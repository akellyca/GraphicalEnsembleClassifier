
#Use this file to run a series of simulations. Recommended to run
#on a server.
# Please ensure that all necessary datasets have been generated prior
# to using this file (via generate_data.py).

models=(nb lr plr gec)
seeds=(0,1,2)
train_ns=(50 75 100 200 300 500 750 1000 2000 3000)

### Artificial simulations
dim=40
np=2
n_ol=10
for seed in ${seeds[@]}; do
	for i in ${train_ns[@]}; do
		for m in ${models[@]}); do
			python train.py artificial $m $seed $n_train $dim $np $n_ol
			python test.py artificial $m $seed $n_train $dim $np $n_ol
	    done
	done
done

# ### 20-Newsgroup simulations
# pair='ac'
# n_trained=500
# nf=500
# for seed in ${seeds[@]}; do
# 	for i in ${train_ns[@]}; do
# 		for m in ${models[@]}); do
# 			python train.py newsgroups $m $seed $n_train $nf $n_trained
# 			python test.py newsgroups $m $seed $n_train $nf $n_trained
# 	    done
# 	done
# done


# ### MNIST simulations
# n1=1
# n2=7
# scheme=focal
# for seed in ${seeds[@]}; do
# 	for i in ${train_ns[@]}; do
# 		for m in ${models[@]}); do
# 			python train.py newsgroups $m $seed $n_train $n1 $n2 $scheme
# 			python test.py newsgroups $m $seed $n_train $n1 $n2 $scheme
# 	    done
# 	done
# done















