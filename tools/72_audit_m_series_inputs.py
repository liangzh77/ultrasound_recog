"""Read-only audit of frozen M-series inputs and checkpoints; never fit a model.

Outputs are separate from the original experiment. Diagnostic channel removal
and BatchNorm probes are not independently trained candidates or validation of
a replacement model. Patient identifiers and source paths are never exported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

from src.common_paths import PATIENT_MULTIMODAL_EXPERIMENT_DIR as EXP
from src.common_paths import PATIENT_MULTIMODAL_REGISTRY_DIR as REG
from src.common_paths import RAW_DATA_DIR
from src.research_annotations import annotation_geometry_signature, normalize_annotation
from src.research_dataset import load_fold_records, PatientBagDataset
from src.research_metrics import compute_patient_metrics
from src.research_models import create_timm_encoder, MaskedMeanClassifier
from src.research_schema import DIAGNOSIS_CLASSES
from src.research_semantic_contours import (
    load_semantic_contour_manifest, render_category_masks,
    SemanticContourResearchImageDataset, PairedSemanticContourAugmentation,
)
from src.research_semantic_features import build_patient_semantic_features

OUT = EXP / 'reports/input_audit_20261116'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save(name, result):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def shared():
    config = yaml.safe_load((ROOT / 'configs/research/m2_semantic_contour_mean_b2.yaml').read_text(encoding='utf-8'))
    categories = config['semantic_contours']['finding_categories']
    records = sum((load_fold_records(REG, ROOT, 0, split) for split in ('train', 'validation', 'test')), [])
    contours = load_semantic_contour_manifest(
        ROOT / config['data']['semantic_manifest_path'], ROOT,
        config['data']['semantic_manifest_sha256'], {r.image_key for r in records}, categories,
    )
    return config, categories, records, contours


def audit_data():
    started = time.monotonic()
    config, categories, records, contours = shared()
    images = {r['image_key']: r for r in read_csv(REG / 'images.csv') if r['include'] == '1'}
    sources = {r['image_key']: r for r in read_csv(REG / 'private/image_sources.csv')}
    frozen = {(RAW_DATA_DIR / r['relative_path']).resolve(): r['sha256'] for r in read_csv(REG / 'source_files.csv') if r['cohort'] == 'development_labeled'}
    dataset = SemanticContourResearchImageDataset(records, contours, categories)
    checks, problems = Counter(), defaultdict(list)
    support = defaultdict(Counter)
    per_patient = defaultdict(Counter)
    tiny_objects, lost_objects = Counter(), Counter()
    sample_candidates = defaultdict(list)
    digest_people = defaultdict(set)
    for index, record in enumerate(records):
        key = record.image_key
        source = sources[key]
        paths = {name: ROOT / source[name] for name in ('raw_image_path', 'raw_annotation_path', 'normalized_annotation_path')}
        raw = json.loads(paths['raw_annotation_path'].read_text(encoding='utf-8'))
        annotation = json.loads(paths['normalized_annotation_path'].read_text(encoding='utf-8'))
        for name in ('raw_image_path', 'raw_annotation_path'):
            sha = digest(paths[name])
            checks[name + '_hash_checked'] += 1
            if sha != frozen.get(paths[name].resolve()): problems[name + '_hash_mismatch'].append(key)
            if name == 'raw_image_path': digest_people[sha].add(record.person_key)
        if paths['raw_image_path'].stem != paths['raw_annotation_path'].stem:
            problems['paired_stem_mismatch'].append(key)
        # info.name is metadata rather than the loader's matching key.
        name = str(raw.get('info', {}).get('name', ''))
        if name and Path(name.replace('\\', '/')).name != paths['raw_image_path'].name:
            problems['info_name_differs_from_paired_filename'].append(key)
        expected, _ = normalize_annotation(raw, record.diagnosis)
        if expected != annotation: problems['normalized_annotation_mismatch'].append(key)
        if annotation_geometry_signature(raw) != annotation_geometry_signature(annotation):
            problems['normalization_geometry_mismatch'].append(key)
        with Image.open(paths['raw_image_path']) as image:
            width, height = image.size
            orientation = image.getexif().get(274, 1)
        if orientation not in (None, 1): problems['nonidentity_exif_orientation'].append(key)
        if (width, height) != (int(annotation['info']['width']), int(annotation['info']['height'])):
            problems['image_annotation_dimensions_mismatch'].append(key)
        if (width, height) != (int(images[key]['width']), int(images[key]['height'])):
            problems['image_registry_dimensions_mismatch'].append(key)
        if any(float(raw['ultrasound_rect'][n]) != record.roi[n] for n in ('x1', 'y1', 'x2', 'y2')):
            problems['roi_registry_mismatch'].append(key)
        masks, counts, pixels = render_category_masks(annotation, record.roi, categories, 384)
        tensor = dataset.load_tensor(record)
        if tensor.shape != (16, 384, 384) or not torch.isfinite(tensor).all():
            problems['invalid_actual_tensor'].append(key)
        tensor_positive = bool(tensor[3:].any())
        support[record.diagnosis]['images'] += 1
        support[record.diagnosis]['positive_images'] += tensor_positive
        per_patient[record.person_key]['images'] += 1
        per_patient[record.person_key]['positive_images'] += tensor_positive
        per_patient[record.person_key]['diagnosis_id'] = record.diagnosis_id
        for ci, category in enumerate(categories):
            stored = contours[key].category_masks.get(category)
            expected_pixels = np.asarray(masks[category]) > 0
            if not np.array_equal(tensor[3 + ci].numpy().astype(bool), expected_pixels):
                problems['actual_tensor_mask_mismatch'].append(key)
            if bool(stored) != bool(counts[category] and pixels[category]):
                problems['sparse_channel_presence_mismatch'].append(key)
            if stored:
                checks['positive_masks_checked'] += 1
                if stored.object_count != counts[category] or stored.positive_pixels != pixels[category]:
                    problems['stored_count_or_area_mismatch'].append(key)
                sample_candidates[category].append((stored.positive_pixels, record))
        # Independently examine each polygon before and after scaling, including
        # objects lost by nearest-neighbour resize or clipping at the ROI.
        left, top = max(0, math.floor(record.roi['x1'])), max(0, math.floor(record.roi['y1']))
        right, bottom = min(width, math.ceil(record.roi['x2'])), min(height, math.ceil(record.roi['y2']))
        scale = 384 / max(right - left, bottom - top)
        dims = (round((right-left)*scale), round((bottom-top)*scale))
        for obj in annotation.get('objects', []):
            cat = obj.get('category')
            if cat not in categories: continue
            points = obj.get('segmentation', [])
            if len(points) < 3: continue
            full = Image.new('L', (width, height))
            ImageDraw.Draw(full).polygon([tuple(p[:2]) for p in points], fill=255)
            cropped = full.crop((left, top, right, bottom))
            before, after = np.count_nonzero(cropped), np.count_nonzero(cropped.resize(dims, Image.Resampling.NEAREST))
            checks['polygons_checked'] += 1
            if any(not (left <= p[0] <= right and top <= p[1] <= bottom) for p in points):
                checks['polygons_with_vertex_outside_roi'] += 1
            if before == 0: checks['polygons_empty_after_roi_crop'] += 1
            if before and not after: lost_objects[cat] += 1
            if after <= 16: tiny_objects[cat] += 1
        checks['images_checked'] += 1
        if (index + 1) % 500 == 0: print('DATA', index + 1, '/', len(records), flush=True)
    # Verify the patient-level feature table really uses this same manifest.
    folds = {r['person_key']: r['outer_fold'] for r in read_csv(REG / 'folds_outer.csv')}
    image_rows = [dict(r, outer_fold=folds[r['person_key']]) for r in images.values()]
    table = build_patient_semantic_features(image_rows, contours, categories, 384*384)
    saved = {r['person_key']: r for r in read_csv(EXP / 'reports/semantic_contours/M1_patient_features.csv')}
    feature_error = max(abs(table.features[i, j] - float(saved[k][n])) for i, k in enumerate(table.person_keys) for j, n in enumerate(table.feature_names))
    for record in records:
        if record.person_key in per_patient:
            p = per_patient.pop(record.person_key)
            support[record.diagnosis]['patients'] += 1
            support[record.diagnosis]['all_zero_patients'] += not p['positive_images']
            support[record.diagnosis]['patients_over_six_images'] += p['images'] > 6
    # Synthetic geometric control: same mask in all channels; independent RGB
    # marker has matching support, with tolerances only for bilinear edges.
    marker = Image.new('L', (384, 384))
    ImageDraw.Draw(marker).rectangle((105, 93, 278, 301), fill=255)
    rgb = Image.merge('RGB', (marker, marker, marker))
    geometry_iou, channel_equal = [], []
    for seed in range(64):
        torch.manual_seed(seed)
        transformed, channels = PairedSemanticContourAugmentation((0, 0, 0))(rgb, [marker.copy() for _ in categories])
        first = np.asarray(channels[0]) > 0
        visible = np.asarray(transformed).mean(2) > 127
        geometry_iou.append(float(np.logical_and(first, visible).sum() / np.logical_or(first, visible).sum()))
        channel_equal.append(all(np.array_equal(first, np.asarray(m) > 0) for m in channels))
    # Deterministic median-size example for each category, cropped to effective
    # view. Private visual evidence stays inside the ignored experiment folder.
    OUT.mkdir(parents=True, exist_ok=True)
    visual_keys = []
    for ci, cat in enumerate(categories):
        choices = sorted(sample_candidates[cat], key=lambda x: (x[0], x[1].image_key))
        _, rec = choices[len(choices)//2]
        t = dataset.load_tensor(rec)
        rgb_array = (t[:3] * torch.tensor([.229,.224,.225])[:,None,None] + torch.tensor([.485,.456,.406])[:,None,None]).clamp(0, 1)
        rgb_array = (rgb_array.permute(1,2,0).numpy()*255).astype(np.uint8)
        overlay = rgb_array.copy()
        foreground = t[3+ci].numpy() > 0
        overlay[foreground] = (overlay[foreground]*.55 + np.array([255,60,20])*.45).astype(np.uint8)
        panel = Image.new('RGB', (768, 410), 'white')
        panel.paste(Image.fromarray(rgb_array), (0, 26)); panel.paste(Image.fromarray(overlay), (384, 26))
        ImageDraw.Draw(panel).text((8, 5), f'Category {ci:02d} | raw ROI / model channel overlay', fill='black')
        panel.save(OUT / f'private_category_{ci:02d}.png')
        visual_keys.append({'category': cat, 'image_key': rec.image_key})
    result = {
        'scope': 'historical_development_only_no_2026_reference_access',
        'checks': dict(checks), 'problem_counts': {k: len(v) for k, v in problems.items()},
        'problem_example_image_keys': {k: v[:5] for k, v in problems.items()},
        'duplicate_exact_image_hashes_across_people': sum(len(v)>1 for v in digest_people.values()),
        'support_by_diagnosis': dict(support), 'm1_feature_max_absolute_error': feature_error,
        'polygons_lost_by_resize': dict(lost_objects), 'polygons_at_most_16_pixels': dict(tiny_objects),
        'synthetic_augmentation_trials': 64, 'minimum_rgb_mask_iou': min(geometry_iou),
        'all_semantic_channels_share_geometry': all(channel_equal),
        'visual_examples': visual_keys, 'seconds': time.monotonic()-started,
    }
    save('data_audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def audit_models():
    config, categories, records, contours = shared()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline = {r['person_key']: r for r in read_csv(EXP / 'reports/oof/M2_oof.csv')}
    probability_columns = ['prob_normal','prob_ra','prob_ga','prob_spa','prob_oa','prob_injury']
    pooled = defaultdict(list)
    fold_results, targets, stored_probabilities, normalization_results = [], [], [], []
    for fold in range(5):
        started = time.monotonic()
        summary = json.loads((EXP / f'reports/M2-fold{fold}-seed{20260724+fold}-formal_summary.json').read_text(encoding='utf-8'))
        checkpoint_path = ROOT / summary['best_checkpoint_path']
        assert digest(checkpoint_path) == summary['best_checkpoint_sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        encoder, dim = create_timm_encoder(config['model']['name'], pretrained=False, in_chans=16)
        model = MaskedMeanClassifier(encoder, dim, 6, .3).to(device).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        fold_records = load_fold_records(REG, ROOT, fold, 'test')
        ds = SemanticContourResearchImageDataset(fold_records, {r.image_key: contours[r.image_key] for r in fold_records}, categories)
        bags = PatientBagDataset(ds, training=False)
        fold_probabilities = defaultdict(list)
        for i in range(len(bags)):
            bag = bags[i]
            inputs = bag['images'].unsqueeze(0).to(device)
            valid = torch.ones(inputs.shape[:2], dtype=torch.bool, device=device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                for mode in ('original', 'zero_semantic_channels', 'zero_normalized_rgb'):
                    changed = inputs if mode=='original' else inputs.clone()
                    if mode == 'zero_semantic_channels': changed[:,:,3:] = 0
                    if mode == 'zero_normalized_rgb': changed[:,:,:3] = 0
                    values = model(changed, valid, instance_chunk_size=6)['patient_probabilities'][0].float().cpu().numpy()
                    pooled[mode].append(values); fold_probabilities[mode].append(values)
            targets.append(bag['target'])
            stored_probabilities.append([float(baseline[bag['person_key']][c]) for c in probability_columns])
            if (i+1)%50==0: print('MODEL fold',fold,'patients',i+1,'/',len(bags),flush=True)
        history = json.loads((EXP / f'logs/M2/fold_{fold}_formal_history.json').read_text(encoding='utf-8'))
        best_row = next(r for r in history if r['epoch']==checkpoint['epoch'])
        weights = checkpoint['model']['encoder.conv_stem.weight']
        fold_results.append({
            'fold': fold, 'checkpoint_sha256': digest(checkpoint_path), 'stem_shape': list(weights.shape),
            'stem_norm_by_input_channel': weights.square().sum((0,2,3)).sqrt().tolist(),
            'parameters': sum(p.numel() for p in model.parameters()),
            'first_train_loss': history[0]['train_loss'], 'last_train_loss': history[-1]['train_loss'],
            'best_epoch': checkpoint['epoch'], 'best_epoch_train_loss': best_row['train_loss'],
            'best_epoch_validation_loss': best_row['validation_loss'],
            'last_validation_loss': history[-1]['validation_loss'],
            'best_validation_macro_f1': checkpoint['validation_macro_f1'],
            'seconds': time.monotonic()-started,
        })
        normalization_results.append(normalization_probe(model, device, fold, categories, contours))
        save('batchnorm_all_folds_diagnostic.json', normalization_results)
        del model, checkpoint
        if device.type=='cuda': torch.cuda.empty_cache()
        print('MODEL fold',fold,'complete',flush=True)
    y = np.asarray(targets)
    probabilities = {m: np.asarray(p) for m,p in pooled.items()}
    original = probabilities['original']
    result = {
        'scope': 'frozen_checkpoint_inference_only_no_fitting_or_selection',
        'removal_interpretation': 'off_distribution_sensitivity_not_retrained_ablation_or_causal_effect',
        'folds': fold_results,
        'normalization_diagnostic': normalization_results,
        'replayed_patients': len(targets),
        'oof_max_probability_absolute_error': float(np.abs(original-np.asarray(stored_probabilities)).max()),
        'oof_argmax_agreement': float((original.argmax(1)==np.asarray(stored_probabilities).argmax(1)).mean()),
        'metrics': {m: compute_patient_metrics(y,p,DIAGNOSIS_CLASSES) for m,p in probabilities.items()},
        'removal_effects': {m:{'changed_top1':int((p.argmax(1)!=original.argmax(1)).sum()),
            'mean_patient_l1_probability_change':float(np.abs(p-original).sum(1).mean())}
            for m,p in probabilities.items() if m!='original'},
    }
    save('model_audit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','folds','normalization_diagnostic')}, ensure_ascii=False, indent=2),flush=True)
    print('METRICS', {m: {k:v for k,v in values.items() if k in ('accuracy','macro_f1')} for m,values in result['metrics'].items()},flush=True)


def audit_diagnostics():
    """Independent rasterization and a prespecified fold-0 normalization probe."""
    config, categories, records, contours = shared()
    sources = {r['image_key']: r for r in read_csv(REG / 'private/image_sources.csv')}
    ious, difference_pixels, reviewed = [], [], 0
    for rec in records:
        if not contours[rec.image_key].category_masks: continue
        raw = json.loads((ROOT / sources[rec.image_key]['raw_annotation_path']).read_text(encoding='utf-8'))
        ann, _ = normalize_annotation(raw, rec.diagnosis)
        w, h = int(ann['info']['width']), int(ann['info']['height'])
        rect = rec.roi
        l, t, r, b = max(0,math.floor(rect['x1'])), max(0,math.floor(rect['y1'])), min(w,math.ceil(rect['x2'])), min(h,math.ceil(rect['y2']))
        scale = 384/max(r-l,b-t)
        rw, rh = round((r-l)*scale), round((b-t)*scale)
        for category, entry in contours[rec.image_key].category_masks.items():
            # Draw at original image coordinates first, then crop. This differs
            # from the production path, which translates polygons before drawing.
            full = Image.new('L',(w,h))
            draw = ImageDraw.Draw(full)
            for obj in ann.get('objects',[]):
                if obj.get('category')==category and len(obj.get('segmentation',[]))>=3:
                    draw.polygon([tuple(p[:2]) for p in obj['segmentation']],fill=255)
            resized = full.crop((l,t,r,b)).resize((rw,rh),Image.Resampling.NEAREST)
            canvas = np.zeros((384,384),dtype=bool)
            canvas[(384-rh)//2:(384-rh)//2+rh,(384-rw)//2:(384-rw)//2+rw] = np.asarray(resized)>0
            with Image.open(entry.path) as im: actual = np.asarray(im)>0
            union = np.logical_or(canvas,actual).sum()
            ious.append(float(np.logical_and(canvas,actual).sum()/union))
            difference_pixels.append(int(np.logical_xor(canvas,actual).sum()))
        reviewed += 1
        if reviewed % 500 == 0: print('INDEPENDENT raster images', reviewed,flush=True)
    raster = {'positive_masks':len(ious),'exact_matches':sum(v==0 for v in difference_pixels),
              'minimum_iou':min(ious),'mean_iou':float(np.mean(ious)),
              'maximum_pixel_difference':max(difference_pixels)}
    save('independent_raster_audit.json',raster)
    print('RASTER',raster,flush=True)
    # One engineering diagnostic, confined to fold 0 train/inner validation.
    # No labels drive normalization, no weights/stats are saved or fitted,
    # dropout stays off, and each patient uses the same <=6 input images in both modes.
    checkpoint = torch.load(EXP/'artifacts/M2/fold_0/best_formal.pt',map_location='cpu',weights_only=False)
    encoder,dim = create_timm_encoder(config['model']['name'],pretrained=False,in_chans=16)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MaskedMeanClassifier(encoder,dim,6,.3).to(device).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    save('batchnorm_diagnostic.json',normalization_probe(model,device,0,categories,contours))


def normalization_probe(model,device,fold,categories,contours):
    model.eval()
    bn_layers=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    for layer in bn_layers: layer.momentum=0.0
    results={}
    for split in ('train','validation'):
        selected=load_fold_records(REG,ROOT,fold,split)
        ds=SemanticContourResearchImageDataset(selected,{r.image_key:contours[r.image_key] for r in selected},categories)
        bags=PatientBagDataset(ds,training=False)
        class_count=Counter(); ys=[]; predictions=defaultdict(list)
        for i in range(len(bags)):
            if split=='train' and class_count[bags.labels[i]]>=12: continue
            bag=bags[i]; class_count[bag['target']]+=1
            inputs=bag['images'][:6].unsqueeze(0).to(device)
            valid=torch.ones(inputs.shape[:2],dtype=torch.bool,device=device)
            with torch.inference_mode(),torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                for mode in ('stored_running_statistics','per_patient_batch_statistics'):
                    for layer in bn_layers: layer.train(mode=='per_patient_batch_statistics')
                    predictions[mode].append(model(inputs,valid)['patient_probabilities'][0].float().cpu().numpy())
            ys.append(bag['target'])
        results[split]={'patients':len(ys),'metrics':{mode:compute_patient_metrics(np.array(ys),np.array(p),DIAGNOSIS_CLASSES) for mode,p in predictions.items()}}
        print('BN PROBE',fold,split,{mode:{k:metrics[k] for k in ('accuracy','macro_f1')} for mode,metrics in results[split]['metrics'].items()},flush=True)
    return {'fold':fold,'scope':'engineering_sensitivity_inner_validation_not_a_replacement_model','batchnorm_layers':len(bn_layers),'images_per_patient':'first_6_or_all_if_fewer','results':results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('data','models','diagnostics'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.phase == 'data': audit_data()
    elif args.phase == 'models': audit_models()
    else: audit_diagnostics()
