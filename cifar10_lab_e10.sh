#!/bin/bash
alpha_length=3
csd_length=3
epoch_length=2
py_number=3
#weights_length=2

hp='--data=mnist --n_round=300 --n_client=50 --actiavte_rate 0.2 --pruing_p=0 --i_seed=10001 --split=True'
alpha=('0.05' '0.1' '1')
csd=('1' '10' '100')
epoch_num=('1' '3')
#weights=('0' '1')
paras[0]='./fedavg.py'
paras[1]='./fola.py'
paras[2]='./fola_prior.py'

for((a=0; a<alpha_length; a++))
    do
        {
        for((e=0; e<epoch_length; e++))
            do
                {
                 for((p=0; p<py_number; p++))
                        do
                            {
                                if [ $p == 0 ]
                                then
                                   echo "Run AVG No Csd"
                                    python ${paras[p]} ${hp} '--alpha='${alpha[a]} '--n_epoch='${epoch_num[e]} '--csd_importance=0' '--weight=1'
                                elif [ $p == 1 ]
                                then
                                   echo "Run FOLA with Csd"
                                   for((c=0; c<csd_length; c++))
                                        do
                                            {
                                                python ${paras[p]} ${hp} '--alpha='${alpha[a]} '--n_epoch='${epoch_num[e]} '--csd_importance='${csd[c]} '--weight=1'
                                            }
                                   done
                                else
                                   echo "Run FOLA_prior with Csd"
                                   for((c=0; c<csd_length; c++))
                                        do
                                            {
                                                python ${paras[p]} ${hp}  '--alpha='${alpha[a]} '--n_epoch='${epoch_num[e]} '--csd_importance='${csd[c]} '--weight=0'
                                            }
                                   done
                                fi

                            }
                    done
                }
        done
    }&
done
wait