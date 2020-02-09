# Defining class 'Employee'
class Employee:
    empCount = 0  # variable assigned (data member) to count the employees
    totalSalary = 0

    # constructor to initialize name, family, salary, department
    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        self.__class__.empCount += 1
        self.__class__.totalSalary = Employee.totalSalary + salary

    # Create function for average salary
    def avg_salary(self):
        avg_Salary = (Employee.totalSalary / Employee.empCount)
        return avg_Salary

# Create instances for passing employee data
emp1 = Employee('Andy', 'T', 3000, 'IT')
emp2 = Employee('Brett', 'W', 4000, 'Finance')
emp3 = Employee('Jondy', 'A', 5000, 'Development')

# Defining sub-class for full time employee inheriting Employee class functions
class FulltimeEmp(Employee):
    FTEcount = 0
    FTEtotalSalary = 0

    def __init__(self, name, family, salary, department):
        Employee.__init__(self, name, family, salary, department)
        self.__class__.FTEcount += 1
        self.__class__.FTEtotalSalary = FulltimeEmp.FTEtotalSalary + salary

    # Creating a member function associated with full time employees class
    def fulltimeCount(self):
        fulltimeCount = FulltimeEmp.FTEcount
        return fulltimeCount

    # Create function to call average salary
    def avgFTEsalary(self):
        avgFTEsalary = (FulltimeEmp.FTEtotalSalary / FulltimeEmp.FTEcount)
        return avgFTEsalary

# Create instances (passing arguments) of the full time employee and calling their member functions
emp4 = FulltimeEmp('Kristin', 'Y', 6000, 'Product')
emp5 = FulltimeEmp('Dale', 'W', 6000, 'PMO')

print(emp1.name)
print(emp2.name)
print(emp3.name)
print(emp4.name)
print(emp5.name)
(emp3.avg_salary())
print("Average Salary of Consultant (Not Full-time)", emp3.avg_salary())
(emp4.avgFTEsalary())
print("Average Salary of Full time Employee", emp4.avgFTEsalary())

# Calling full time employee function to identify the number of employees
totalEmp = emp5.fulltimeCount() + emp1.empCount
print("Total no. of Employees in the Company are ", totalEmp)
