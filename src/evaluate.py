import os
from tqdm import tqdm

from eval import *
import matplotlib.pyplot as plt 

run_dir = "../models"
task = "cot_2nn"
  
run_ids = [
]
method_name = []
step3_res = []
step3_ood_res = []
step2_res = []
step2_ood_res = []
step1_res = []
step1_ood_res = []
for run_id in tqdm(run_ids):
    run_path = os.path.join(run_dir, task, run_id)
    step3_metric, step2_metric, step1_metric = get_run_metrics(run_path)
    step3_ood_metric, step2_ood_metric, step1_ood_metric = get_run_metrics_ood(run_path)
    step3_res.append(step3_metric['mean'])
    step3_ood_res.append(step3_ood_metric['mean'])
    
    step2_res.append(step2_metric['mean'])
    step2_ood_res.append(step2_ood_metric['mean'])
    
    step1_res.append(step1_metric['mean'])
    step1_ood_res.append(step1_ood_metric['mean'])

res_lis = [step1_res, step2_res, step3_res]
ood_res_lis = [step1_ood_res, step2_ood_res, step3_ood_res]
print('DONE!') 

plt.clf()
fig, axs = plt.subplots(1,3,figsize=(15,4))
for j in range(3):
    for i in range(len(res_lis[0])): 
        risk = res_lis[j][i] 
        if j == 0:
            axs[j].plot(risk, label=method_name[i])
        else:
            axs[j].plot(risk )
    axs[j].set_xlabel('# ICL Examples')
    axs[j].set_ylabel('Task risk')
    axs[j].set_title(f'Step {j+1}')
handles, labels = [], []
for ax in axs.flat: 
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

fig.legend(handles, labels, loc='right' )
 
plt.tight_layout()
plt.subplots_adjust(right=0.78)  

plt.savefig('test.pdf')


plt.clf()
fig, axs = plt.subplots(1,3,figsize=(15,4))
for j in range(3):
    for i in range(len(ood_res_lis[0])):
        risk = ood_res_lis[j][i]
        if j == 0:
            axs[j].plot(risk, label=method_name[i])
        else:
            axs[j].plot(risk )
    axs[j].set_xlabel('# ICL Examples')
    axs[j].set_ylabel('Task risk')
    axs[j].set_title(f'Step {j+1}')

handles, labels = [], []
for ax in axs.flat: 
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

fig.legend(handles, labels, loc='right' )
 
plt.tight_layout()
plt.subplots_adjust(right=0.78)  
 
plt.savefig('test_ood.pdf')
