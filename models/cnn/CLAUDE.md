## 帮我解决下面的bug，并且之后要一直运行代码进行测试直到没有bug为止

(base) PS C:\Users\26747\Desktop\ml_pro_master\fluid_model> python train.py --config configs/quick_test_cnn.json
2025-08-31 19:06:58,688 - __main__ - INFO - Starting Fluid Dynamics Model Training
2025-08-31 19:06:58,689 - __main__ - INFO - Loading config from: configs/quick_test_cnn.json
2025-08-31 19:06:58,702 - __main__ - INFO - ============================================================
2025-08-31 19:06:58,703 - __main__ - INFO - TRAINING CONFIGURATION
2025-08-31 19:06:58,703 - __main__ - INFO - ============================================================
2025-08-31 19:06:58,703 - __main__ - INFO - Model: FluidCNN
2025-08-31 19:06:58,703 - __main__ - INFO - Model config: configs/models/cnn_nano.json
Traceback (most recent call last):
  File "C:\Users\26747\Desktop\ml_pro_master\fluid_model\train.py", line 184, in <module>
    main()
    ~~~~^^
  File "C:\Users\26747\Desktop\ml_pro_master\fluid_model\train.py", line 96, in main
    logger.info(f"Model size: d_model={model_config.d_model}, n_heads={model_config.n_heads}, n_layers={model_config.n_layers}")
                                       ^^^^^^^^^^^^^^^^^^^^
AttributeError: 'CNNConfig' object has no attribute 'd_model'