import numpy as np
import onnxruntime as ort
import torch

import bc_resnet_model
import get_data

pt_file_name = "model-sc-2-test.pt"
onnx_file_name = "model-sc-2-test.pt.onnx"
# Load the ONNX model
ort_session = ort.InferenceSession(onnx_file_name)
test_tensor = np.random.randn(1, 1, 40, 32).astype(np.float32)
outputs = ort_session.run(
    None,
    {"input": test_tensor},
)

model = bc_resnet_model.BcResNetModel(
    n_class=get_data.N_CLASS,
    scale=2,
    dropout=0.1,
    use_subspectral=True,
)
model.load_state_dict(torch.load(pt_file_name))
model.eval()

print(outputs[0])
print(model(torch.tensor(test_tensor)))
