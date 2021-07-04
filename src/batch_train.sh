python train.py --mode train --model lstm --epochs 100 > ../result/lstm_200e.log
python train.py --mode train --model ltc --epochs 100 > ../result/ltc_200e.log
python train.py --mode train --model node --epochs 100 > ../result/node_200e.log
python train.py --mode train --model ctgru --epochs 100 > ../result/ctgru_200e.log
python train.py --mode train --model ctrnn --epochs 100 > ../result/ctrnn_200e.log
# python train.py --model ltc_rk --epochs 200 > ../result/ltc_rk_200e.log
# python train.py --model ltc_ex --epochs 200 > ../result/ltc_ex_200e.log