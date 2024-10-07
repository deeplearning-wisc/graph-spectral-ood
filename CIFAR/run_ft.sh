#$1 must belong to:
#  graph_ood, scone, woods, woods_nn, energy, energy_vos, OE
#
#$2 must belong to:
#  cifar10, cifar100, imagenet100, MNIST
#
#$3 must belong to:
#  svhn, lsun_c, lsun_r,
#  isun, dtd, places, FashionMNIST, tinyimages_300k
#
#$4 must belong to:
#  svhn, lsun_c, lsun_r,
#  isun, dtd, places, FashionMNIST, tinyimages_300k

learning_rate=0.005
batch_size=512
ngpu=1
prefetch=4
epochs=20
script=train.py
gpu=3
seed=100

eta=-10.0

alpha=0.50

cortype='gaussian_noise'
# snapshots/pretrained
load_pretrained='./snapshots/svhn/l0.50_u2.00_epoch_1000.pt'
checkpoints_dir='./path/to/save/checkpoints'
results_dir="results"
name="supspectral"
gamma_l=0.0200
gamma_u=2.0

pi_1=0.5
pi_2=0.1

score="$1"
dataset="$2"
aux_out_dataset="$3"
test_out_dataset="$4"

echo "running $score with dataset $dataset, aux_out_dataset $aux_out_dataset test_out_dataset $test_out_dataset, pi_1=$pi_1, pi_2=$pi_2 on GPU $gpu"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
--model wrn --score "$1" --learning_rate $learning_rate --epochs $epochs \
--test_out_dataset "$4" --aux_out_dataset "$3" --batch_size=$batch_size --ngpu=$ngpu \
--prefetch=$prefetch  --results_dir=$results_dir \
--checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained \
--name=$name --gamma_l=$gamma_l --gamma_u=$gamma_u \
--pi_1=$pi_1 --seed=$seed \
--pi_2=$pi_2 \
--eta=$eta \
--alpha=$alpha \
--cortype=$cortype
