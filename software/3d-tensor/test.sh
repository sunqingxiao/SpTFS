OP=('train' 'test')
for((i=0;i<3;i++))
do
    for((j=0;j<5;j++))
    do
        python Dl3dNet.py ${OP[0]} data/ft_mul_mode${i}_cv${j}_train.npz data/ft_mul_mode${i}_cv${j}_test.npz data/mode${i}_cv${j}_WrongIds.npz
        python Dl3dNet.py ${OP[1]} data/ft_mul_mode${i}_cv${j}_train.npz data/ft_mul_mode${i}_cv${j}_test.npz data/mode${i}_cv${j}_WrongIds.npz
        sleep 3
    done
done
