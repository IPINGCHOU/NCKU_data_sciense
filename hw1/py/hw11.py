
# coding: utf-8

# In[2]:


#import candy crush board data
#file_name = "E:/WORKOUT/Statistic/data_sciense_intro/hw1/candy_input1.txt"
file_name = "E:/WORKOUT/Statistic/data_sciense_intro/hw1/candy_input2.txt"
myfile = open(file_name, "r")

candy_1 = []
for line in myfile:
    candy_1.append(line.strip())
myfile.close()

for i in range(len(candy_1)):
    candy_1[i] = candy_1[i].split(',')

#test if the data imported successfully or not
for i in range(len(candy_1)):
    print(candy_1[i])


# In[3]:


stable_switch = 0

#continue the loop until the board is stable
while stable_switch == 0:
    hori_move = 0
    vert_move = 0
    lined_x = []
    lined_y = []

# horiziontal pair check
# if there's at least 3 same numbers lining up horiziointally
# store the coordinate in lined_x and lined_y
# switch hori_move to 1 if this part has been activated
    for i in range(len(candy_1)):
        for j in range(len(candy_1[1])-2):
            if candy_1[i][j] != 0:
                if candy_1[i][j] == candy_1[i][j+1] == candy_1[i][j+2]:
                    lined_x.append(j)
                    lined_x.append(j+1)
                    lined_x.append(j+2)
                    lined_y.extend((i,i,i))
                    hori_move = 1
# vertical pair check
# if there's at least 3 same numbers lining up vertically
# store the coordinate in lined_x and lined_y
# switch vert_move to 1 if this part has been activated
    for i in range(len(candy_1)-2):
        for j in range(len(candy_1[1])):
            if candy_1[i][j] != 0:
                if candy_1[i][j] == candy_1[i+1][j] == candy_1[i+2][j]:
                    lined_x.extend((j,j,j))
                    lined_y.append(i)
                    lined_y.append(i+1)
                    lined_y.append(i+2)
                    vert_move = 1
# if both horiziontal pair check and vertical pair check
# didn't discover any lined up numbers in the board (both hori_move and vert_move == 0)
# then the board is stable
# time to stop the loop
    if hori_move == 0 and vert_move == 0:
        break

# set each recorded coordinate in lined_y and lined_x to 0
# stands for "crushed"
    for k in range(len(lined_x)):
        candy_1[lined_y[k]][lined_x[k]] = 0

# if there's "crushed" candy under the "uncrushed" candy
# level down all the "uncrushed candy" in correspond column once
# replace 0 at the top of the "uncrushed" candy
# repeat the move until all the "uncrushed" candy reach the bottom of the board
# or there's no "crushed" candy under it
    for i in range(len(candy_1[1])):
        for j in range(len(candy_1)):
            if candy_1[j][i] == 0:
                for k in range(0,j,1):
                    if(k < j):
                        candy_1[j-k][i] = candy_1[j-k-1][i]
                candy_1[0][i] = 0;


# In[4]:


#test output
for i in range(len(candy_1)):
    print(candy_1[i])


# In[8]:


#output the candy_1 file as "candy_output.txt"
f = open('candy_output.txt', 'w', encoding = 'UTF-8') 
f.write(str(candy_1))

