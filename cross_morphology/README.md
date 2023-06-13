## collect data with a random policy from both domains
```shell
# collect source domain data
python collect_data.py --data_type 'base' --data_id 1 --env HalfCheetah-v2
# collect target domain data
python collect_data.py --data_type '3leg' --data_id 1 --env HalfCheetah_3leg-v2
```

## obtain the policy in the source domain
```shell
cd base_train_test/td3_solver
python train.py 
```

## train the correspondence
```shell
cd cycle_transfer
python alignexp.py
```