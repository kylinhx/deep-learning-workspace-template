## utils.py

```python
# project setup
setup_seed(seed)
freeze_model(torch.nn.Module: model)

# model info
calculate_model_params(torch.nn.Module: model)
calculate_gflops(torch.nn.Module: model, tuple: input_shape)
get_model_structure(torch.nn.Module: model, tuple: input_shape)
get_infer_time(torch.nn.Module: model, torch.utils.data.DataLoader: test_loader)

# data io
read_yaml(str: file_path)

# sys info
get_system_info()

# plot result figure
plot_confusion_matrix(numpy.ndarray: y_true, numpy.ndarray: y_pred, list: labels, str: title, str: saved_path, bool: normalize)
plot_cruve(str: saved_path, list: data_list, list: data_labels)

# save or load model
save_model_with_extra_info(model, saved_path)
save_model(model, saved_path)
load_model_with_extra_info(model_path)
load_model(model_path)
```