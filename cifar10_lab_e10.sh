#!/bin/bash
alpha_length=3
csd_length=3
hp='--data=cifar10 --n_round=500 --n_client=50 --actiavte_rate 0.2 --pruing_p=0'
alpha=('0.05' '0.1' '1')
paras[0]='./fedavg.py'
paras[1]='./fola.py'
paras[2]='./fola_prior.py'
py_number=3
csd=('1' '10' '100')
epoch = ('1' '3')
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
                                    python ${paras[p]} ${hp} '--weight='${weights[e]} '--alpha='${alpha[a]} '--csd_importance=0'
                                elif [ $p == 1 ]
                                then
                                   echo "Run FOLA with Csd"
                                   for((c=0; c<csd_length; c++))
                                        do
                                            {
                                                python ${paras[p]} ${hp} '--weight='${weights[e]} '--alpha='${alpha[a]} '--csd_importance='${csd_ours[c]}
                                            }
                                   done
                                else
                                   echo "Run FOLA_prior with Csd"
                                   for((c=0; c<csd_length; c++))
                                        do
                                            {
                                                python ${paras[p]} ${hp} '--weight='${weights[e]} '--alpha='${alpha[a]} '--csd_importance='${csd_ours[c]}
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