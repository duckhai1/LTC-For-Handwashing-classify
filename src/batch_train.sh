EPOCH_NUM=50
# python train.py --mode train --model lstm --epochs ${EPOCH_NUM}
python train.py --mode train --model ltc --epochs ${EPOCH_NUM} 
python train.py --mode train --model node --epochs ${EPOCH_NUM}
python train.py --mode train --model ctgru --epochs ${EPOCH_NUM}
python train.py --mode train --model ctrnn --epochs ${EPOCH_NUM}
# python train.py --model ltc_rk --epochs 200 > ../result/ltc_rk_200e.log
# python train.py --model ltc_ex --epochs 200 > ../result/ltc_ex_200e.log