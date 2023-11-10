import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import math
from scipy.special import xlogy
from tqdm import tqdm


def entropy(dist):
    ## Ignore zero values in distributions
    return -np.sum(dist*np.log2(dist, where=(dist!=0)), axis=0)


def optimization(pyx, ub=1, p =0, q=0):
    ##### for P(Y=p|do(X=q)) #####
    nx = pyx.shape[1]
    ny = pyx.shape[0]
    px = pyx.sum(axis=0)
    
    # uy_x: 1x(nx*ny) vector
    uy_x = cp.Variable(nx*ny)
    v1 = np.zeros(nx*ny)
    v1[p*nx:p*nx+nx] = px
    pydox = uy_x @ v1

    v2 = np.zeros((ny, nx*ny))
    for i in range(ny):
        v2[i, i*nx:(i+1)*nx] = px
    # qy: 1xny vector 
    qy = uy_x @ v2.T
    # qyx: 1x(nx*ny) vector
    qyx = qy @ v2

    v3 = np.zeros((nx*ny))
    for i in range(ny):
        v3[i*nx:(i+1)*nx] = px

    uyx = cp.multiply(uy_x, v3)
    dkl = cp.kl_div(uyx, qyx)/math.log(2)
    # I = dkl @ np.ones((nx*ny))
    I = cp.sum(dkl)

    
    v4 = np.zeros((nx*ny, ny))
    for i in range(ny):
        v4[nx*i+q, i] = px[q]
    
    v5 = np.zeros((nx*ny, nx))
    for i in range(nx):
        v5[i:nx*ny:nx, i] = 1
    constraints  = [uy_x @ v4 == pyx[:,q],
                     uy_x@v5 == np.ones(nx), 
                     uy_x >= 0, uy_x <= 1,
                     I <= ub]
    
    max_obj = cp.Maximize(pydox)
    min_obj = cp.Minimize(pydox)
    

    max_prob = cp.Problem(max_obj, constraints)
    max_prob.solve(solver=cp.SCS)
    # print("The upper bound of p(y0|do(x0)) is\n", max_prob.value)
    min_prob = cp.Problem(min_obj, constraints)
    min_prob.solve(solver=cp.SCS)
    # print("The lower bound of p(y0|do(x0)) is\n", min_prob.value)
    # print(f"The optimization bound is\n [{min_prob.value}, {max_prob.value}]")
    return min_prob.value, max_prob.value


### read a text file with the data
data = np.loadtxt('adult.data', delimiter=',', dtype=str)
# print(data)
## take all the second column from the data

data_numeric = np.zeros((len(data), len(data[0])), dtype=int)


## get rid of the spaces 
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i,j] = data[i,j].strip()


## make a dictionary for attributes 

## 2nd column
workclass = {'Private':0, 'Self-emp-not-inc':1, 'Self-emp-inc':2, 'Federal-gov':3, 'Local-gov':4, 'State-gov':5, 'Without-pay':6, 'Never-worked':7, '?':-1}

## 3rd column: fnlwgt

## 4th column
education = {'Bachelors':0, 'Some-college':1, '11th':2, 'HS-grad':3, 'Prof-school':4, 'Assoc-acdm':5, 'Assoc-voc':6, '9th':7, '7th-8th':8, '12th':9, 'Masters':10, '1st-4th':11, '10th':12, 'Doctorate':13, '5th-6th':14, 'Preschool':15}

## 6th column
marital_status = {'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2, 'Separated':3, 'Widowed':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6}

## 7th column
occupation = {'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3, 'Exec-managerial':4, 'Prof-specialty':5, 'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8, 'Farming-fishing':9, 'Transport-moving':10, 'Priv-house-serv':11, 'Protective-serv':12, 'Armed-Forces':13, '?':-1}

## 8th column
relationship = {'Wife':0, 'Own-child':1, 'Husband':2, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':5}

## 9th column
race = {'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3, 'Black':4}
## 10th column
sex = {'Female':0, 'Male':1}

## 14th column
native_country = {'United-States':0, 'Cambodia':1, 'England':2, 'Puerto-Rico':3, 'Canada':4, 'Germany':5, 'Outlying-US(Guam-USVI-etc)':6, 'India':7, 'Japan':8, 'Greece':9, 'South':10, 'China':11, 'Cuba':12, 'Iran':13, 'Honduras':14, 'Philippines':15, 'Italy':16, 'Poland':17, 'Jamaica':18, 'Vietnam':19, 'Mexico':20, 'Portugal':21, 'Ireland':22, 'France':23, 'Dominican-Republic':24, 'Laos':25, 'Ecuador':26, 'Taiwan':27, 'Haiti':28, 'Columbia':29, 'Hungary':30, 'Guatemala':31, 'Nicaragua':32, 'Scotland':33, 'Thailand':34, 'Yugoslavia':35, 'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38, 'Hong':39, 'Holand-Netherlands':40, '?':-1}

## 15th column
income = {'<=50K':0, '>50K':1}


## read the data and convert the attributes to numbers
for i in range(len(data)):
    data_numeric[i,0] = int(data[i,0])
    data_numeric[i,1] = int(workclass[data[i,1]])
    data_numeric[i,2] = int(data[i,2])
    data_numeric[i,3] = int(education[data[i,3]])
    data_numeric[i,4] = int(data[i,4])
    data_numeric[i,5] = int(marital_status[data[i,5]])
    data_numeric[i,6] = int(occupation[data[i,6]])
    data_numeric[i,7] = int(relationship[data[i,7]])
    data_numeric[i,8] = int(race[data[i,8]])
    data_numeric[i,9] = int(sex[data[i,9]])
    data_numeric[i,10] = int(data[i,10])
    data_numeric[i,11] = int(data[i,11])
    data_numeric[i,12] = int(data[i,12])
    data_numeric[i,13] = int(native_country[data[i,13]])
    data_numeric[i,14] = int(income[data[i,14]])
    



def get_distributions(data, idx, shape, dist_path, pz):
    
    outcome_i, treatment_i, confounder_i = idx 
    outcome_shape, treatment_shape, confounder_shape  = shape
    py_xz_path, px_z_path, pz_path = dist_path
    print(py_xz_path)
    py_xz = np.zeros([outcome_shape, treatment_shape, confounder_shape])
    px_z = np.zeros([treatment_shape, confounder_shape])

    for i in range(py_xz.shape[2]):
        z = np.copy(data)
        z = z[z[:,treatment_i] != -1,:]
        z_idx = z[:,confounder_i] == i
        z = z[z_idx,:]
        z_count = np.sum(z_idx)
        if z_count == 0:
            print(f"no data for confounder {i}")
            continue
        
        for j in range(py_xz.shape[1]):
            x_idx = z[:,treatment_i] == j
            x = z[x_idx,:]
            x_count = np.sum(x[:,outcome_i]!= -1)
            
            if x_count == 0:
                print(f"no data for treatment {i},{j}")
                continue
            px_z[j,i] = x_count/z_count
            
            for k in range(py_xz.shape[0]):
                y_idx = x[:,outcome_i] == k
                y_count = np.sum(y_idx)
                # if y_count == 0:
                #     print(f"no data for outcome {i},{j},{k}")
                #     continue
                py_xz[k,j,i] = y_count/x_count
    
    print(py_xz.shape)
    print(py_xz.sum())
    print("")

    print(px_z.shape)
    print(px_z.sum(axis=0))
    print("")


    ## save the probability data to a numpy file
    np.save(f'{py_xz_path}', py_xz)
    np.save(f'{px_z_path}', px_z)
    np.save(f'{pz_path}', pz)
    



## simplify marital status as married and not married
data_numeric[data_numeric[:,5]>=5,5] = 0
data_numeric[data_numeric[:,5]>=1,5] = 1
## show histogram of marital status


## Simplify the 8th feature as mainlander and non-mainlander
## [0.95, 0.0497]
data_numeric[data_numeric[:,8]==4, 8] = 0
data_numeric[data_numeric[:,8]!=0, 8] = 1
pRac = np.zeros(2)
for i in range(pRac.shape[0]):
    pRac[i] = np.sum(data_numeric[:,8]==i)/np.sum(data_numeric[:,8]!=-1)

## simplify the education as above high school and below high school
data_numeric[data_numeric[:,3]==1,3] = 0
data_numeric[data_numeric[:,3]==3,3] = 0
data_numeric[data_numeric[:,3]==4,3] = 0
data_numeric[data_numeric[:,3]==5,3] = 0
data_numeric[data_numeric[:,3]==6,3] = 0
data_numeric[data_numeric[:,3]==10,3] = 0
data_numeric[data_numeric[:,3]==13,3] = 0
data_numeric[data_numeric[:,3]!=0,3] = 1
pEdu = np.zeros(2)
for i in range(pEdu.shape[0]):
    pEdu[i] = np.sum(data_numeric[:,3]==i)/np.sum(data_numeric[:,3]!=-1)


## simplify the age as young and old

data_numeric[data_numeric[:,0]<=65,0] = 0
data_numeric[data_numeric[:,0]>65,0] = 1

## get distribution of age
pAge = np.zeros(2)
for i in range(pAge.shape[0]):
    pAge[i] = np.sum(data_numeric[:,0]==i)/np.sum(data_numeric[:,0]!=-1)


## simplify relationship as spouse and not spouse
data_numeric[data_numeric[:,7]==2,7] = 0
data_numeric[data_numeric[:,7]!=0,7] = 1

# Simplify hours as part-time and full-time

treatment_idx = 12
data_numeric[data_numeric[:,treatment_idx]<40,treatment_idx] = 0
data_numeric[data_numeric[:,treatment_idx]>=40,treatment_idx] = 1
pHours = np.zeros(2)
for i in range(pHours.shape[0]):
    pHours[i] = np.sum(data_numeric[:,treatment_idx]==i)/np.sum(data_numeric[:,treatment_idx]!=-1)

pEdu_Age = np.zeros((2,2))
for j in range(pEdu_Age.shape[1]):
    Age_count = np.sum(data_numeric[:,0]==j)
    age_idx = data_numeric[:,0]==j
    age_data = data_numeric[age_idx,:]
    for i in range(pEdu_Age.shape[0]):
        pEdu_Age[i,j] = np.sum(age_data[:,3]==i)/Age_count
print(f"Conditional distribution of education given age:\n {pEdu_Age}")

pHour_Age = np.zeros((2,2))
for j in range(pHour_Age.shape[1]):
    Age_count = np.sum(data_numeric[:,0]==j)
    age_idx = data_numeric[:,0]==j
    age_data = data_numeric[age_idx,:]
    for i in range(pHour_Age.shape[0]):
        pHour_Age[i,j] = np.sum(age_data[:,12]==i)/Age_count



pRelation_HourEduAge = np.zeros((2,2,2,2))

for k in range(pRelation_HourEduAge.shape[3]):
    Age_count = np.sum(data_numeric[:,0]==k)
    age_idx = data_numeric[:,0]==k
    age_data = data_numeric[age_idx,:]
    for j in range(pRelation_HourEduAge.shape[2]):
        Edu_count = np.sum(age_data[:,3]==j)
        edu_idx = age_data[:,3]==j
        edu_data = age_data[edu_idx,:]
        for i in range(pRelation_HourEduAge.shape[1]):
            Hour_count = np.sum(edu_data[:,12]==i)
            hour_idx = edu_data[:,12]==i
            hour_data = edu_data[hour_idx,:]
            for l in range(pRelation_HourEduAge.shape[0]):
                pRelation_HourEduAge[l,i,j,k] = np.sum(hour_data[:,7]==l)/Hour_count

print(pRelation_HourEduAge.sum())


pIncome_RelationHourEduAge = np.zeros((2,2,2,2,2))

for k in range(pIncome_RelationHourEduAge.shape[4]):
    Age_count = np.sum(data_numeric[:,0]==k)
    age_idx = data_numeric[:,0]==k
    age_data = data_numeric[age_idx,:]
    for j in range(pIncome_RelationHourEduAge.shape[3]):
        Edu_count = np.sum(age_data[:,3]==j)
        edu_idx = age_data[:,3]==j
        edu_data = age_data[edu_idx,:]
        for i in range(pIncome_RelationHourEduAge.shape[2]):
            Hour_count = np.sum(edu_data[:,12]==i)
            hour_idx = edu_data[:,12]==i
            hour_data = edu_data[hour_idx,:]
            for l in range(pIncome_RelationHourEduAge.shape[1]):
                Relation_count = np.sum(hour_data[:,7]==l)
                relation_idx = hour_data[:,7]==l
                relation_data = hour_data[relation_idx,:]
                for m in range(pIncome_RelationHourEduAge.shape[0]):
                    pIncome_RelationHourEduAge[m,l,i,j,k] = np.sum(relation_data[:,14]==m)/Relation_count


    
eps = 1e-3
        
        
pIncomeRelationHourEduAge = np.einsum('ijklm, jklm, km, lm, m -> ijklm', pIncome_RelationHourEduAge, pRelation_HourEduAge, pHour_Age, pEdu_Age, pAge)

entropy_age = entropy(pAge)
entropy_edu = entropy(pEdu)
entropy_hour = entropy(pHours)
print(f"Entropy of age: {entropy_age}")
print(f"Entropy of education: {entropy_edu}")
print(f"Entropy of hours: {entropy_hour}")

pRelationEdu = pIncomeRelationHourEduAge.sum(axis=(0,2,4))


pIncomeEdu = pIncomeRelationHourEduAge.sum(axis=(1,3,4))

pIncomeHour = pIncomeRelationHourEduAge.sum(axis=(1,2,4))

pHourEdu = pIncomeRelationHourEduAge.sum(axis=(0,1,4))

pIncomeRelationHourEdu = pIncomeRelationHourEduAge.sum(axis=4)
pIncomeRelation_HourEdu = np.einsum('ijkl, kl -> ijkl', pIncomeRelationHourEdu, 1/pHourEdu)

i_set = ['<=50K', '>50K']
j_set = ['family', 'Not in family']
k_set = ['part-time', 'full-time']
l_set = ['above high school ', 'below high school']

# print(entropy_age)
pHourEduAge = pIncomeRelationHourEduAge.sum(axis=(0,1))
pAge_HourEdu = np.einsum('ijk, ij -> kij', pHourEduAge, 1/pHourEdu)
# print(pAge_HourEdu.sum(axis=0))
c_entropy_age = entropy(pAge_HourEdu)
# print(c_entropy_age)

for i in range(pIncomeRelation_HourEdu.shape[0]):
    for j in range(pIncomeRelation_HourEdu.shape[1]):
        for k in range(pIncomeRelation_HourEdu.shape[2]):
            for l in range(pIncomeRelation_HourEdu.shape[3]):
                entropy_age = c_entropy_age[k, l]
                
                lb, ub = optimization(pIncomeRelation_HourEdu[:,:,k, l], entropy_age, i, j)
                neg = np.arange(pIncomeRelation_HourEdu.shape[0]) != i
                tp_lb, tp_ub = pIncomeRelation_HourEdu[i,j,k,l], 1-pIncomeRelation_HourEdu[neg,j,k,l].sum()
                # print(f"Entropy Constraint bounds \n [{lb, ub}]")
                # print(f"Tian-Pearl bounds\n [{tp_lb, tp_ub}]")
                # print("")
                if ((lb - tp_lb) > eps) or ((ub - tp_ub) < -eps):
                    print(entropy_age)
                    print(f"p(Income={i_set[i]}, Relation={j_set[j]}| Hour={k_set[k]}, Edu={l_set[l]})")
                    print(f"Entropy Constraint bounds \n [{lb, ub}]")
                    print(f"Tian-Pearl bounds\n [{tp_lb, tp_ub}]")
                    print("")

