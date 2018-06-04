
# coding: utf-8

# In[9]:


def answer31():
#data import
    import csv
    file_name = "E:/WORKOUT/Statistic/data_sciense_intro/hw1/census.csv"
    census = open(file_name, 'r')
    csvCursor = csv.reader(census)
    data = []
    for row in csvCursor:
        data.append(row)
    census.close()

    all_stn = []
    counts = []
    max_count_pos = []
    ans = []

#store all STNAME in list all_stn
    for i in range(1, len(data), 1):
        all_stn.append(data[i][5])

#count duplicate times of each state
#return the most duplicate times state
    my_dict = {i:all_stn.count(i) for i in all_stn}
    counts = list(my_dict.values())
    max_count_pos = counts.index(max(counts))
    ans = list(my_dict.keys())
    return(ans[max_count_pos])

answer31()


# In[8]:


def answer32():
#import data
    import csv
    file_name = "E:/WORKOUT/Statistic/data_sciense_intro/hw1/census.csv"
    census = open(file_name, 'r')
    csvCursor = csv.reader(census)
    data = []
    for row in csvCursor:
        data.append(row)
    census.close()
    
    all_names = []
    all_stn = []
    ans = []
#store all STNAME in all_stn
    for i in range(1, len(data), 1):
        all_stn.append(data[i][5])

#count duplicate times of each state
    my_dict = {i:all_stn.count(i) for i in all_stn}
    all_names = list(my_dict.keys())
    all_dup = list(my_dict.values())
    all_pop = [0]*len(all_names)
    steps = 1
    ans = []

#only caculate the most populous counties for each state
    for i in range(0, len(all_names), 1):
        temp_pop_list = []
#store all population data of each state in temp_pop_list
        for j in range(all_dup[i]):
            temp_pop_list.append(data[steps][7])
            steps = steps + 1
#pick the 3 most populous counties and add them
#store the result in list all_pop
        for k in range(0,3,1):
            temp_pop_list = list(map(int,temp_pop_list))
            if(temp_pop_list != []):
                temp_pop = []
                temp_pop = temp_pop_list.index(max(temp_pop_list))
                all_pop[i] = all_pop[i] + int(temp_pop_list.pop(temp_pop))
#pick the 3 most populous state for list all_pop
#store them in list ans
#return ans
    for m in range(0,3,1):
        temp_ans_pop = []
        temp_ans_pop = all_pop.index(max(all_pop))
        ans.append(all_names.pop(temp_ans_pop))
        all_pop.pop(temp_ans_pop)
        
    return(ans)
    
answer32()
    


# In[23]:


def answer33():
#import data
    import csv
    file_name = "E:/WORKOUT/Statistic/data_sciense_intro/hw1/census.csv"
    census = open(file_name, 'r')
    csvCursor = csv.reader(census)
    data = []
    for row in csvCursor:
        data.append(row)
    census.close()

#store POPESTIMATE2010 ~ POPESTIMATE2015 by each county in list temp_six
    counts = [0]*len(data)
    for i in range(1, len(data), 1):
        temp_six = [0]*6
        for j in range(0,6,1):
            temp_six[j] = data[i][9+j]
#find the max and min in temp_six and subtract them
#store the result in list counts
        counts[i] = abs(int(max(temp_six)) - int(min(temp_six)))
#find the max one in list counts
#return ans
    ans = data[counts.index(max(counts))][6]
    return(ans)

answer33()

