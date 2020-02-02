#Convert pound into Kilograms

N = int(input("Enter the number of Students = "))
list = []
for i in range(N): #loop statement for rage N
    Wlb = int(input("Enter weight in Lbs = "))
    Wkg = float(Wlb*0.453592) #Converting the weight of students in the class from lbs to kgs
    list.append(Wkg)
print(list)