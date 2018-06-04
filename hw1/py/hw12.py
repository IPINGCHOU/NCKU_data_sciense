
# coding: utf-8

# In[20]:


def answer21():
#import data
    import csv
    file_name = "E:/WORKOUT/Statistic/data_sciense_intro/hw1/olympics.csv"
    olympics = open(file_name, 'r')
    csvCursor = csv.reader(olympics)
    data = []
    for row in csvCursor:
        data.append(row)
    olympics.close()

    num_summer_metal = []
    max_index = []
    max_country = []
#store all summer gold metals numbers of each country in list num_summer_metal
    for i in range(2, len(data),1):
        num_summer_metal.append(data[i][2])

#find the max one in list num_summer_metal
#return ans
    max_index = num_summer_metal.index(max(num_summer_metal))
    max_country = data[max_index+2][0]
    return(max_country)
    
answer21()


# In[22]:


def answer22():
#import data
    import csv
    file_name = "E:/WORKOUT/Statistic/data_sciense_intro/hw1/olympics.csv"
    olympics = open(file_name, 'r')
    csvCursor = csv.reader(olympics)
    data = []
    for row in csvCursor:
        data.append(row)
    olympics.close()
    
    num_metal =[]
    max_index = []
    max_country = []

#subtract summer gold metal counts with winter gold metal counts of each country
#store them in list num_metal
    for i in range(2, len(data)-1,1):
        num_metal.append(abs(int(data[i][2]) - int(data[i][7])))
#find the max one in list num_metal
    max_index = num_metal.index(max(num_metal))
    max_country = data[max_index+2][0]
    return(max_country)

answer22()


# In[29]:


def answer23():
#import data
    import csv
    file_name = "E:/WORKOUT/Statistic/data_sciense_intro/hw1/olympics.csv"
    olympics = open(file_name, 'r')
    csvCursor = csv.reader(olympics)
    data = []
    for row in csvCursor:
        data.append(row)
    olympics.close()
    
    num_metal =[]
    max_index = []
    max_country = []
#if the country has both at least one metal in summer and winter olympics
#subtract summer gold metal counts with winter gold metal counts of each country
#divide them by total gold metal counts of each country
#store the result in num_metal
#if the country don't
#store 0 in the list instead
    for i in range(2, len(data)-1, 1):
        if int(data[i][2]) > 0 and int(data[i][7]) > 0:
            num_metal.append(abs(int(data[i][2]) - int(data[i][7])) / int(data[i][12]))
        else:
            num_metal.append(0)
#find the max one in list num_metal
#return the ans
    max_index = num_metal.index(max(num_metal))
    max_country = data[max_index+2][0]
    return(max_country)

answer23()
    

