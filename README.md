User/Admin Manual
Please run them seperately and run the clients one at a time if training. 

To train federated learning on model:

py server_yolo.py

py client_yolo.py --server localhost:8080 --model static/output/final_model.pt --labeled_dir data/labelfront1 --work_root output/client1 --epochs 3 --nc 1 --names object

py client_yolo.py --server localhost:8080 --model static/output/final_model.pt --labeled_dir data/labelfront2 --work_root output/client2 --epochs 3 --nc 1 --names object

py client_yolo.py --server localhost:8080 --model static/output/final_model.pt --labeled_dir data/labelback1 --work_root output/client3 --epochs 3 --nc 1 --names object

py client_yolo.py --server localhost:8080 --model static/output/final_model.pt --labeled_dir data/labelback2 --work_root output/client4 --epochs 3 --nc 1 --names object

To test against federated learning final model:

python server_yolo.py --test_only
