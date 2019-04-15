import os
import sys
import copy
import time
import random
import pyspark
from statistics import mean
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import desc, size, max, abs,row_number, monotonically_increasing_id
import numpy as np

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def toCSVLineRDD(rdd):
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row])) \
        .reduce(lambda x,y: os.linesep.join([x,y]))
    return a + os.linesep

def toCSVLine(data):
    if isinstance(data, RDD):
        if data.count() > 0:
            return toCSVLineRDD(data)
        else:
            return ""
    elif isinstance(data, DataFrame):
        if data.count() > 0:
            return toCSVLineRDD(data.rdd)
        else:
            return ""
    return None

def kmeans1(filename,cluster_number,seed):

    def distance_cal(dict_a,dict_b):
        return sum([(dict_a[key]-dict_b[key])**2 for key in dict_a])

    #
    def nearest_centroid(x,dict_x,y):
        min_distance=0
        near_centroid=str()
        for i in range(0,len(y)):
            if i==0 or  min_distance > distance_cal(dict_x,y[i][1]):
                near_centroid = str(y[i][0])
                min_distance = distance_cal(dict_x,y[i][1])
        return near_centroid


    def dist(x,y):
        # for key in x:
        #     if key in y:
        #         x[key] = (x[key] + y[key])
        return {key: x[key]+y[key] for key in x}

    def calculate_centroid(cluster,RDD):

        output=[]
        for i in range(0,len(cluster)):
            RDD1=RDD.filter(lambda x: x[0] in cluster[i][1])
            m=RDD1.count()
            RDD1 = RDD1.map(lambda x:(cluster[i][0],x[1])).reduceByKey(lambda x,y:dist(x,y))
            RDD1= RDD1.collect()[0]
            for key in RDD1[1]:
                RDD1[1][key] = RDD1[1][key]/m
            output=output+[RDD1]
        return output
    '''
    initial points are here 
    '''
    spark = init_spark()
    inputRDD = spark.read.text(filename).rdd
    inputRDD = inputRDD.map(lambda x: x.value.split(","))
    cells_name1 = inputRDD.first()
    cells_name = cells_name1[1:]
    inputRDD= inputRDD.filter(lambda x: x != cells_name1)
    input=inputRDD.map(lambda x:x[0])
    gene_name=input.collect()[1:]
    inputRDD = inputRDD.map(lambda x: (x[1:],x[0]))
    inputRDD = inputRDD.flatMap(lambda x: [(cells_name[i] if int(x[0][i])!=0 else "null1",[x[1]]) for i in range(0,len(x[0]))]).filter(lambda x: x[0]!="null1")
    inputRDD = inputRDD.reduceByKey(lambda x,y: x+y)
    inputRDD = inputRDD.map(lambda x: (x[0],set(x[1])))
    inputRDD = inputRDD.map(lambda x: (x[0],[(gene_name[i],1 if gene_name[i] in x[1] else 0) for i in range (0,len(gene_name))]))
    inputRDD = inputRDD.map(lambda x: (x[0],dict(x[1])))
    random.seed(seed)
    init_centers = random.sample(cells_name, cluster_number)
    inputRDD1 = inputRDD.map(lambda x: (x[0],x[1],init_centers))
    cells_vector = inputRDD1.filter(lambda x:  x[0] in x[2]).map(lambda x: (x[0],x[1])).collect()
    cluster = inputRDD.map(lambda x: (nearest_centroid(x[0],x[1],cells_vector),[x[0]])).reduceByKey(lambda x,y:x+y)
    cluster= cluster.map(lambda x:(x[0],sorted(x[1])))
    flag = 0
    s=0
    temp=cluster.collect()

    while flag==0:
        ''' [(name of init state,dictinary vector of cluster's centroids)]=new_centroid'''
        new_centroid = calculate_centroid(temp,inputRDD)
        cluster = inputRDD.map(lambda x: (nearest_centroid(x[0],x[1],new_centroid),[x[0]])).reduceByKey(lambda x,y:x+y)
        cluster= cluster.map(lambda x:(x[0],sorted(x[1])))
        s=s+1
        a= cluster.collect()

        if temp == a or s==3:
            flag=1
        else:
            temp = a
    output = a
    output = dict(output)
    for key, value in output.items():
        value.sort()
    output=list(output.values())
    return output


def kmeans_jaccard(filename,cluster_number,seed):

    def distance_cal(dict_a,dict_b):
        intersection = sum([(dict_a[key] and dict_b[key]) for key in dict_a])
        union = sum([(dict_a[key] or dict_b[key]) for key in dict_a])
        return 1 - int(intersection/union)

    #

    def nearest_centroid(x,dict_x,y):
        min_distance=0
        near_centroid=str()
        for i in range(0,len(y)):
            if i==0 or  min_distance > distance_cal(dict_x,y[i][1]):
                near_centroid = str(y[i][0])
                min_distance = distance_cal(dict_x,y[i][1])
        return near_centroid

    def calculate_centroid(cluster,RDD):
        output=[]
        for i in range(0,len(cluster)):
            RDD1 = RDD.filter(lambda x: x[0] in cluster[i][1])
            RDD = RDD.map(lambda x: x[1])
            state = RDD1.collect()
            min = 0
            centroid= {}
            for j in range(0,len(state)):
                s=sum([distance_cal(state[j],state[k])^2 for k in range(0,len(state))])
                if min==0 or min>s:
                    min = s
                    centroid = state[j]
            output = output+[(cluster[i][0],centroid)]
        return output


    '''
    initial points are here 
    '''


    spark = init_spark()
    inputRDD = spark.read.text(filename).rdd
    inputRDD = inputRDD.map(lambda x: x.value.split(","))
    cells_name1 = inputRDD.first()
    cells_name = cells_name1[1:]
    inputRDD= inputRDD.filter(lambda x: x != cells_name1)
    input=inputRDD.map(lambda x:x[0])
    gene_name=input.collect()[1:]
    inputRDD = inputRDD.map(lambda x: (x[1:],x[0]))
    inputRDD = inputRDD.flatMap(lambda x: [(cells_name[i] if int(x[0][i])!=0 else "null1",[x[1]]) for i in range(0,len(x[0]))]).filter(lambda x: x[0]!="null1")
    inputRDD = inputRDD.reduceByKey(lambda x,y: x+y)
    inputRDD = inputRDD.map(lambda x: (x[0],set(x[1])))
    inputRDD = inputRDD.map(lambda x: (x[0],[(gene_name[i],1 if gene_name[i] in x[1] else 0) for i in range (0,len(gene_name))]))
    inputRDD = inputRDD.map(lambda x: (x[0],dict(x[1])))
    random.seed(seed)
    init_centers = random.sample(cells_name, cluster_number)
    inputRDD1 = inputRDD.map(lambda x: (x[0],x[1],init_centers))
    cells_vector = inputRDD1.filter(lambda x:  x[0] in x[2]).map(lambda x: (x[0],x[1])).collect()
    cluster = inputRDD.map(lambda x: (nearest_centroid(x[0],x[1],cells_vector),[x[0]])).reduceByKey(lambda x,y:x+y)
    cluster= cluster.map(lambda x:(x[0],sorted(x[1])))
    flag = 0
    s=0
    temp=cluster.collect()

    while flag==0:
        ''' [(name of init state,dictinary vector of cluster's centroids)]=new_centroid'''
        new_centroid = calculate_centroid(temp,inputRDD)
        cluster = inputRDD.map(lambda x: (nearest_centroid(x[0],x[1],new_centroid),[x[0]])).reduceByKey(lambda x,y:x+y)
        cluster= cluster.map(lambda x:(x[0],sorted(x[1])))
        s=s+1
        a= cluster.collect()
        if temp == a and s==1:
            flag=1
        else:
            temp = a
    output  = a
    output = dict(output)
    for key, value in output.items():
        value.sort()
    output=list(output.values())
    return output
def kmeans_Euclidean_non_binary(filename,cluster_number,seed):

    def distance_cal(dict_a,dict_b):
        # final_dist = 0
        # for key in dict_a:
        #     final_dist += (dict_a[key] - dict_b[key])**2
        return sum([(dict_a[key]-dict_b[key])**2 for key in dict_a])

    #
    def nearest_centroid(x,dict_x,y):
        min_distance=0
        near_centroid=str()
        for i in range(0,len(y)):
            if i==0 or  min_distance > distance_cal(dict_x,y[i][1]):
                near_centroid = str(y[i][0])
                min_distance = distance_cal(dict_x,y[i][1])
        return near_centroid


    def dist(x,y):

        return {key: x[key]+y[key] for key in x}

    def calculate_centroid(cluster,RDD):

        output=[]
        for i in range(0,len(cluster)):
            RDD1=RDD.filter(lambda x: x[0] in cluster[i][1])
            m=RDD1.count()
            RDD1 = RDD1.map(lambda x:(cluster[i][0],x[1])).reduceByKey(lambda x,y:dist(x,y))
            RDD1= RDD1.collect()[0]
            for key in RDD1[1]:
                RDD1[1][key] = RDD1[1][key]/m
            output=output+[RDD1]
        return output
    '''
    initial points are here 
    '''
    spark = init_spark()
    inputRDD = spark.read.text(filename).rdd
    inputRDD = inputRDD.map(lambda x: x.value.split(","))
    cells_name1 = inputRDD.first()
    cells_name = cells_name1[1:]
    inputRDD= inputRDD.filter(lambda x: x != cells_name1)
    input=inputRDD.map(lambda x:x[0])
    gene_name=input.collect()[1:]
    # x is list of tuples and output is list of tuples (gene name, read depth)
    def dictionary_maker(x,gene_name):
        out=[]
        for i in range(0,len(gene_name)):
            s=0
            for j in range(0,len(x)):
                if gene_name[i] in x[j]:
                    out = out + [(gene_name[i],int(x[j][1]))]
                    s=1
            if s==0:
                out=out+[(gene_name[i],0)]
        return out
    random.seed(seed)
    init_centers = random.sample(cells_name, cluster_number)
    print('bbbbbb',init_centers)

    inputRDD = inputRDD.map(lambda x: (x[1:],x[0]))
    inputRDD = inputRDD.flatMap(lambda x: [(cells_name[i] if int(x[0][i])!=0 else "null1",[(x[1],int(x[0][i]))]) for i in range(0,len(x[0]))]).filter(lambda x: x[0]!="null1")
    inputRDD = inputRDD.reduceByKey(lambda x,y: x+y)
    inputRDD = inputRDD.map(lambda x: (x[0],dictionary_maker(x[1],gene_name)))
    inputRDD = inputRDD.map(lambda x: (x[0],dict(x[1])))
    inputRD = inputRDD.collect()
    print('aaaaaa',inputRD, len(inputRD))

    inputRDD1 = inputRDD.map(lambda x: (x[0],x[1],init_centers))
    cells_vector = inputRDD1.filter(lambda x:  x[0] in x[2]).map(lambda x: (x[0],x[1])).collect()
    cluster = inputRDD.map(lambda x: (nearest_centroid(x[0],x[1],cells_vector),[x[0]])).reduceByKey(lambda x,y:x+y)
    cluster= cluster.map(lambda x:(x[0],sorted(x[1])))
    flag = 0
    s=0
    temp=cluster.collect()

    while flag==0:
        ''' [(name of init state,dictinary vector of cluster's centroids)]=new_centroid'''
        new_centroid = calculate_centroid(temp,inputRDD)
        cluster = inputRDD.map(lambda x: (nearest_centroid(x[0],x[1],new_centroid),[x[0]])).reduceByKey(lambda x,y:x+y)
        cluster= cluster.map(lambda x:(x[0],sorted(x[1])))
        s=s+1
        a= cluster.collect()

        if temp == a or s==1:
            flag=1
        else:
            temp = a
    output = a
    output = dict(output)
    for key, value in output.items():
        value.sort()
    output=list(output.values())
    return output





'''this function calulates the confusion matrix of our Kmeans'''
def calculate_accuracy(filename):

    clusters_list = kmeans1(filename,4,123)
    output=[]
    for i in range(0,len(clusters_list)):
        off=0
        low=0
        intermediate=0
        high=0
        for j in range(0,len(clusters_list[i])):
            label=clusters_list[i][j].split("_")
            if label[1]=='off':
                off+=1
            elif label[1] == 'low':
                low+=1
            elif label[1] == 'intermediate':
                intermediate+=1
            else:
                high+=1
        output.append([off,low,intermediate,high])
    print(np.matrix(output))
    return output



def kmeans1_plus(filename,cluster_number,seed):

    def distance_cal(dict_a,dict_b):
        return sum([(dict_a[key]-dict_b[key])**2 for key in dict_a])

    #
    def nearest_centroid(x,dict_x,y):
        min_distance=0
        near_centroid=str()
        for i in range(0,len(y)):
            if i==0 or  min_distance > distance_cal(dict_x,y[i][1]):
                near_centroid = str(y[i][0])
                min_distance = distance_cal(dict_x,y[i][1])
        return near_centroid
    def nearest_centroid1(x,dict_x,y):
        min_distance = 0
        near_centroid = str()
        c={}
        for i in range(0,len(y)):
            if i==0 or  min_distance > distance_cal(dict_x,y[i][1]):
                near_centroid = str(y[i][0])
                c = y[i][1]
                min_distance = distance_cal(dict_x,y[i][1])
        return (near_centroid,c)

    def dist(x,y):
        # for key in x:
        #     if key in y:
        #         x[key] = (x[key] + y[key])
        return {key: x[key]+y[key] for key in x}

    def calculate_centroid(cluster,RDD):

        output=[]
        for i in range(0,len(cluster)):
            RDD1=RDD.filter(lambda x: x[0] in cluster[i][1])
            m=RDD1.count()
            RDD1 = RDD1.map(lambda x:(cluster[i][0],x[1])).reduceByKey(lambda x,y:dist(x,y))
            RDD1= RDD1.collect()[0]
            for key in RDD1[1]:
                RDD1[1][key] = RDD1[1][key]/m
            output=output+[RDD1]
        return output

    def kmeans_mini(inputRDD,cluster_number):
        cells_name = inputRDD.map(lambda x: x[0]).collect()

        random.seed(seed)
        init_centers = random.sample(cells_name, cluster_number)
        inputRDD1 = inputRDD.map(lambda x: (x[0],x[1],init_centers))
        cells_vector = inputRDD1.filter(lambda x:  x[0] in x[2]).map(lambda x: (x[0],x[1])).collect()
        cluster = inputRDD.map(lambda x: (nearest_centroid(x[0],x[1],cells_vector),[x[0]])).reduceByKey(lambda x,y:x+y)
        cluster= cluster.map(lambda x:(x[0],sorted(x[1])))
        flag = 0
        s=0
        temp=cluster.collect()

        while flag==0:
            ''' [(name of init state,dictinary vector of cluster's centroids)]=new_centroid'''
            new_centroid = calculate_centroid(temp,inputRDD)
            cluster = inputRDD.map(lambda x: (nearest_centroid(x[0],x[1],new_centroid),[x[0]])).reduceByKey(lambda x,y:x+y)
            cluster= cluster.map(lambda x:(x[0],sorted(x[1])))
            s=s+1
            a= cluster.collect()

            if temp == a or s==10:
                flag=1
            else:
                temp = a
        return new_centroid

    def multiply_dict(x,y):
        for key in x:
            x[key] *= y
        return x

    '''
    initial points are here 
    '''
    spark = init_spark()
    inputRDD = spark.read.text(filename).rdd
    inputRDD = inputRDD.map(lambda x: x.value.split(","))
    cells_name1 = inputRDD.first()
    cells_name = cells_name1[1:]
    inputRDD= inputRDD.filter(lambda x: x != cells_name1)
    input=inputRDD.map(lambda x:x[0])
    gene_name=input.collect()[1:]
    inputRDD = inputRDD.map(lambda x: (x[1:],x[0]))
    inputRDD = inputRDD.flatMap(lambda x: [(cells_name[i] if int(x[0][i])!=0 else "null1",[x[1]]) for i in range(0,len(x[0]))]).filter(lambda x: x[0]!="null1")
    inputRDD = inputRDD.reduceByKey(lambda x,y: x+y)
    inputRDD = inputRDD.map(lambda x: (x[0],set(x[1])))
    inputRDD = inputRDD.map(lambda x: (x[0],[(gene_name[i],1 if gene_name[i] in x[1] else 0) for i in range (0,len(gene_name))]))
    inputRDD = inputRDD.map(lambda x: (x[0],dict(x[1])))
    random.seed(seed)
    init_centers = random.sample(cells_name, 50)
    # init_centers=['"CD8_off_GCTATTCCTGAA"', '"CD8_low_TTCCCCGCCATA"', '"CD8_intermediate_AAGAGCAAATGA"', '"CD8_high_GCTACTGAATTT"']
    inputRDD1 = inputRDD.map(lambda x: (x[0],x[1],init_centers))
    cells_vector1 = inputRDD1.filter(lambda x:  x[0] in x[2]).map(lambda x: (x[0],x[1]))
    cells_vector = cells_vector1.collect()
    cluster = inputRDD.map(lambda x: (nearest_centroid1(x[0],x[1],cells_vector),[x[0]])).reduceByKey(lambda x,y:x+y)
    cluster_weight = cluster.map(lambda x:(x[0][0],multiply_dict(x[0][1],len(x[1]))))
    # list of tuples([(name of cluster,weight)])and out put is the [('aa',dict of new vectors)]
    cells_vector = kmeans_mini(cluster_weight,cluster_number)
    cluster = inputRDD.map(lambda x: (nearest_centroid(x[0],x[1],cells_vector),[x[0]])).reduceByKey(lambda x,y:x+y)
    cluster= cluster.map(lambda x:(x[0],sorted(x[1])))
    flag = 0
    s=0
    temp=cluster.collect()

    while flag == 0:
        ''' [(name of init state,dictinary vector of cluster's centroids)]=new_centroid'''
        new_centroid = calculate_centroid(temp,inputRDD)
        cluster = inputRDD.map(lambda x: (nearest_centroid(x[0],x[1],new_centroid),[x[0]])).reduceByKey(lambda x,y:x+y)
        cluster= cluster.map(lambda x:(x[0],sorted(x[1])))
        s=s+1
        a= cluster.collect()

        if temp == a or s==3:
            flag=1
        else:
            temp = a
    output = a
    output = dict(output)
    for key, value in output.items():
        value.sort()
    output=list(output.values())
    return output






