import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info

if __name__ == '__main__':
    model_path = 'xxxxxxx.pt'
    model = RTDETR(model_path)
    result = model.val(data='dataset.yaml', split='test', imgsz=640, batch=8,
                       project='runs/val', name='val_result', exist_ok=True)

    if model.task == 'detect':
        pre, inf, post = result.speed['preprocess'], result.speed['inference'], result.speed['postprocess']
        _, n_p, _, flops = model_info(model.model)
        size_mb = f'{os.path.getsize(model_path) / 1024 / 1024:.1f}'

        info_table = PrettyTable(title="Model Info")
        info_table.field_names = ["GFLOPs", "Parameters", "Preprocess Time/Image", "Inference Time/Image", "Postprocess Time/Image",
                                  "FPS(Pre+Inf+Post)", "FPS(Inference)", "Model File Size"]
        info_table.add_row([f'{flops:.1f}', f'{n_p:,}', f'{pre/1000:.6f}s', f'{inf/1000:.6f}s',
                            f'{post/1000:.6f}s', f'{1000/(pre+inf+post):.2f}', f'{1000/inf:.2f}', f'{size_mb}MB'])
        print(info_table)

        rd = result.results_dict
        box = result.box
        metrics_table = PrettyTable(title="Model Metrice")
        metrics_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx, cls_name in enumerate(result.names.values()):
            metrics_table.add_row([cls_name, f'{box.p[idx]:.4f}', f'{box.r[idx]:.4f}', f'{box.f1[idx]:.4f}',
                                   f'{box.ap50[idx]:.4f}', f'{box.all_ap[idx, 5]:.4f}', f'{box.ap[idx]:.4f}'])
        metrics_table.add_row(["all(mean)", f'{rd["metrics/precision(B)"]:.4f}', f'{rd["metrics/recall(B)"]:.4f}',
                               f'{np.mean(box.f1):.4f}', f'{rd["metrics/mAP50(B)"]:.4f}',
                               f'{np.mean(box.all_ap[:, 5]):.4f}', f'{rd["metrics/mAP50-95(B)"]:.4f}'])
        print(metrics_table)

        with open(result.save_dir / 'paper_data.txt', 'w+') as f:
            f.write(str(info_table) + '\n' + str(metrics_table))