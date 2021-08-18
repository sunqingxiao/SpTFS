import os
import sys
import random

gz_folder = ""
gz_file = ""
ungz_folder = ""

# generate 3-d tensor from matrixes
def getTensor(fm_dir, bm_dir, tensor_dir):
    # 3-d tensor information
    tensor_nnz = 0

    # forward matrix information
    fm_dim0 = 0
    fm_dim1 = 0
    fm_nnz = 0
    fm_count = -2
    
    # backward matrix information
    bm_dim0 = 0
    bm_dim1 = 0
    bm_nnz = 0
    bm_count = -2

    # get forward matrix dim and nnz
    fmfile = open(fm_dir)
    for fmline in fmfile:
        if fmline.find('%') > -1:
            continue    
        else:
            fmseg = fmline.split()
            fm_dim0 = int(fmseg[-2])
            fm_dim1 = int(fmseg[-3])
            fm_nnz = int(fmseg[-1])
            break
    fmfile.close()

    # get backward matrix dim and nnz
    bmfile = open(bm_dir)
    for bmline in bmfile:
        if bmline.find('%') > -1:
            continue
        else:
            bmseg = bmline.split()
            bm_dim0 = int(bmseg[-2])
            bm_dim1 = int(bmseg[-3])
            bm_nnz = int(bmseg[-1])
            break
    bmfile.close()

    # control tensor nnz range
    bm_obtained = int(bm_nnz / bm_dim0)
    if bm_obtained < 1:
        print("second matrix no value can be obtained")
        return -1

    tensor_nnz = bm_obtained * fm_nnz
    #if tensor_nnz < 1000000 or tensor_nnz > 100000000:
    #if tensor_nnz > 100000000:
    if tensor_nnz > 1000000:
        print ("(Out of range) tensor_nnz = {}".format(tensor_nnz))
        return -1
    else:
        print ("Meet conditions: tensor_nnz = {}".format(tensor_nnz))

    # initialze forward matrix and backward matrix
    forward_matrix = [[0 for i in range(2)] for j in range(fm_nnz)]
    backward_matrix = [0 for i in range(bm_obtained)]

    # write matrix file to forward_matrix and backward_matrix
    fmfile = open(fm_dir)
    for fmline in fmfile:
        if fmline.find('%') > -1:
            continue
        else:
            fm_count += 1
            if fm_count > -1:
                fmseg = fmline.split()
                forward_matrix[fm_count][0] = int(fmseg[-2])
                forward_matrix[fm_count][1] = int(fmseg[-3])
    fmfile.close()

    # create folder according to the range of nnz
    counter = 1
    pow_counter = 0
    for i in range(0, 9):
        counter *= 10
        pow_counter += 1
        if tensor_nnz / counter < 1:
            break
    os.system('mkdir -p hicoo-datasets/pow{}_nnz'.format(pow_counter))

    # write to tensor file
    #tensorfile = file(tensor_dir, "a+")
    if os.path.exists('hicoo-datasets/pow{}_nnz/{}'.format(pow_counter, tensor_dir)):
        return -2

    tensorfile = open('hicoo-datasets/pow{}_nnz/{}'.format(pow_counter, tensor_dir), "w+")
    tensorfile.write('3\n'+str(fm_dim0)+' '+str(fm_dim1)+' '+str(bm_dim1)+'\n')

    for i in range(0, fm_nnz):
        for j in range(0, bm_obtained):
            backward_matrix[j] = random.randint(1, bm_dim1)

        backward_matrix.sort()

        for j in range(0, bm_obtained):
            tensorfile.write(str(forward_matrix[i][0])+' '+str(forward_matrix[i][1])+' '+ \
                    str(backward_matrix[j])+' 1.0\n')

    tensorfile.close()
    return 0

matrixes = []

if __name__ == '__main__':
    mt_location = "/home/sqx/tensor-format/datasets/ungz-matrixset/filter_vector-mtx/irre-matrix-all/hasvalue-irre"

    for x in os.listdir(mt_location):
       matrixes.append(x)
   
    for i in range(0,200): 
        fm_matrix = '.'.join(random.sample(matrixes, 1))
        bm_matrix = '.'.join(random.sample(matrixes, 1))

   
        fm_dir = os.path.join(mt_location, fm_matrix)
        bm_dir = os.path.join(mt_location, bm_matrix)

        tensor_dir = fm_matrix.replace('.mtx','')+"."+bm_matrix.replace('.mtx','')+".tns"

        tensor_path = os.path.join("", tensor_dir)

        print("**********  GENERATING %s **********"%(tensor_dir))
        tensorflag = getTensor(fm_dir, bm_dir, tensor_dir)
        if tensorflag == -2:
            print("%s already exists! CONTINUE!"%tensor_dir)
        elif tensorflag == -1:
            print("%s out of range!!!!!!!!!!!!!!!!"%(tensor_dir))
        elif tensorflag == 0:
            print("%s file generate SUCESS!"%(tensor_dir))
        else:
            print("%s unexpected error!!!!!!"%(tensor_dir))
