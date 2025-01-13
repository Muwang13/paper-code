#!/bin/bash
alpha_length=4
csd_length=3
hp='--data=cifar10 --n_round=20 --n_client=20 --pruing_p=0'
alpha=('0.01' '0.05' '0.1' '1')
gpu_num=('1' '2' '3')
time_num='0' #('0' '1' '2' '3' '4')
paras[0]='./fedavg.py'
paras[1]='./fola.py'
paras[2]='./fola_prior.py'
py_number=3
csd_ours=('0' '1' '100')
weights=('0' '1')
weights_length=2

for((a=0; a<alpha_length; a++))
    do
        {
        for((e=0; e<weights_length; e++))
            do
                {
                 for((p=0; p<py_number; p++))
                        do
                            {
                                if [ $p == 0 ]
                                then
                                   echo "Run AVG No Csd"
                                   # python ${paras[p]} ${hp} '--n_epoch='${n_epoch[e]} '--alpha='${alpha[a]} '--csd_importance=0' '--gpu='${gpu_num[a]}
                                elif [ $p == 1 ]
                                then
                                   echo "Run Ours with Csd"
                                   for((c=0; c<csd_length; c++))
                                        do
                                            {
                                                python ${paras[p]} ${hp} '--n_epoch='${n_epoch[e]} '--alpha='${alpha[a]} '--csd_importance='${csd_ours[c]} '--gpu='${gpu_num[a]}
                                            }
                                   done
                                elif [ $p == 2 ]
                                then
                                   echo "Run Curv with Csd"
                                   for((c=0; c<csd_length; c++))
                                        do
                                            {
                                                python ${paras[p]} ${hp} '--n_epoch='${n_epoch[e]} '--alpha='${alpha[a]} '--csd_importance='${csd_curv[c]} '--gpu='${gpu_num[a]}
                                            }
                                   done

                                else
                                   echo "Run Prox with Csd"
                                   for((c=0; c<csd_length; c++))
                                        do
                                            {
                                                python ${paras[p]} ${hp} '--n_epoch='${n_epoch[e]} '--alpha='${alpha[a]} '--csd_importance='${csd_prox[c]} '--gpu='${gpu_num[a]}
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