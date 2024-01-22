# DroneCrowd VID toolkit - python version

This repository contains the Python version of the DroneCrowd VID toolkit. The repository is based on the [DroneCrowd VID toolkit](https://github.com/VisDrone/DroneCrowd/tree/master/STNNet/DroneCrowd-VID-toolkit).


## Requirements

Install the required packages using the following command:

```bash
pip3 install -r requirements.txt
```

## Usage

Run the following command:
```
Usage: eval.py [OPTIONS]

Options:
  -d, --dataset [dronecrowd|upcount]
                                  Dataset name
  -p, --preds PATH                Path to predictions
  -t, --threads INTEGER           Number of threads
  --help                          Show this message and exit.
```

For example:
```bash
python3 eval.py -d dronecrowd -p ../results/pt_pred/dronecrowd/
```


