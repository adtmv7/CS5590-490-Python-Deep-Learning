# Class 1: Flight Booking Reservation System (e.g. classes Flight, Person, Employee, Passenger etc.)

class Flight():

    # INIT CONSTRUCTOR
    def __init__(self, Flight_ID, Flight_Origin, Flight_Destination, Number_of_Stops, Flight_Type, Passenger_ID, Passenger_TYPE):
        self.Flight_ID = Flight_ID
        self.origin = Flight_Origin
        self.destination = Flight_Destination
        self.stops = Number_of_Stops
        self.Flight_Type = Flight_Type
        self.pid = Passenger_ID
        self.ptype = Passenger_TYPE

    def get_Flight_details(self, Flight_ID):
        print("Flight#:", Flight_ID)
        print("Origin:", self.origin)
        print("Destination:", self.destination)
        print("Flight:", self.Flight_Type)

# Class2: Person which contains the personID(Passenger_ID), their name, phone number, gender, type of person(
# employee/passenger) and it inherits the Flight class to get the Flight details.

class Person(Flight):
    # INIT CONSTRUCTOR
    def __init__(self, Passenger_ID, Passenger_TYPE, Passenger_Gender, Passenger_Name, Passenger_Phonenumber, Flight_ID, Flight_Origin, Flight_Destination, Number_of_Stops, Flight_Type):
        self.name = Passenger_Name
        self.gender = Passenger_Gender
        self.Passenger_Phonenumber = Passenger_Phonenumber
# Here we also use super class to use the parameters from Flight class
        super(Person, self).__init__(Flight_ID, Flight_Origin, Flight_Destination, Number_of_Stops, Flight_Type, Passenger_ID, Passenger_TYPE)

class Employee(Person):
    # INIT CONSTRUCTOR
    def __init__(self, Passenger_ID, Passenger_TYPE, Passenger_Gender, Passenger_Name, Passenger_Phonenumber, Flight_ID, Flight_Origin, Flight_Destination, Number_of_Stops, Flight_Type):
        super(Employee,self).__init__(Passenger_ID, Passenger_TYPE, Passenger_Gender, Passenger_Name, Passenger_Phonenumber, Flight_ID, Flight_Origin, Flight_Destination, Number_of_Stops, Flight_Type)

# This method is to get the travel details of the employee
    def get_travel_details_employee(self):
        print("Pilot: ", self.name, "Here are your Flight details")
        print("Flight_ID:", self.Flight_ID)
        print("Origin:", self.origin)
        print("Destination:", self.destination)

# Class 4:Passenger which is an inherited class from Person, Passport Number is the private data member,
# since we cant reveal it.
class Passenger(Person):
    names = []
    d = dict()
    # INIT CONSTRUCTOR

    def __init__(self, Passenger_ID, Passenger_TYPE, Passenger_Gender, Passenger_Name, Passenger_Phonenumber, Flight_ID, Flight_Origin, Flight_Destination, Number_of_Stops, Flight_Type):
        super(Passenger, self).__init__(Passenger_ID, Passenger_TYPE, Passenger_Gender, Passenger_Name, Passenger_Phonenumber, Flight_ID, Flight_Origin, Flight_Destination, Number_of_Stops, Flight_Type)

# This is to get the travellers on the plane into a list, where we have the FlightNumber(Flight_ID)
        # as the key and the passengername(name) as the value.
        if self.Flight_ID in Passenger.d.keys():
            Passenger.d[self.Flight_ID].append(self.name)
        else:
            Passenger.d[self.Flight_ID] = [self.name]

    # This method is to get the travel details of the passenger
    def get_travel_details_passanger(self):
        print("Travel Details of ", self.name)
        print("Flight Id:",   self.Flight_ID)
        print("Flight Type:", self.Flight_Type)
        print("Origin:", self.origin)
        print("Destination:", self.destination)

# This method is to print the dictionary where we have stored the passengers list for different Flights
    def get_travelling_passengers(self):
        print("Passengers on the Flight", Passenger.d)


class Ticket(Passenger):
    def __init__(self, Passenger_ID, Passenger_TYPE, Passenger_Gender, Passenger_Name, Passenger_Phonenumber, Flight_ID, Flight_Origin, Flight_Destination, Number_of_Stops,
                 Flight_Type, boarding_group_no, row, seat_no):
        super(Ticket, self).__init__(Passenger_ID, Passenger_TYPE, Passenger_Gender, Passenger_Name,
                                     Passenger_Phonenumber, Flight_ID, Flight_Origin, Flight_Destination, Number_of_Stops, Flight_Type)
        self.boarding_group_no = boarding_group_no
        self.row = row
        self.seat_no = seat_no
        print("Your ticket details are below: ")

    def get_boarding_pass(self, Passenger_Name):
        for k, v in Passenger.d.items():
            names = v
            for i in names:
                if i == Passenger_Name:
                    print("Passenger Name:", Passenger_Name)
                    print("Flight Id:", k)
                    print("Boarding Group and Seat No:", self.boarding_group_no, self.row, self.seat_no)
                    print("Origin:", self.origin)
                    print("Destination:", self.destination)

# Passenger_ID,Passenger_TYPE,Passenger_Gender,Passenger_Name,Passenger_Phonenumber,Flight_ID,Flight_Origin,Flight_Destination,Number_of_Stops,Flight_Type
#Flight_ID, Flight_Origin, Flight_Destination, Number_of_Stops, Flight_Type, Passenger_ID, Passenger_TYPE
f1 = Flight("EY1234", "ABH", "JFK", 1, "Etihad Airways", "P1", "P4")
f1.get_Flight_details("EY1234")

#Instances of Passenger class
p1 = Passenger("P1", "P", "F", "Andrew S", 4258796358, "F1400", "MCI", "BOM", 3, "Etihad Airways")
p2 = Passenger("P2", "P", "M", "Hannah T", 2588796358, "F1445", "MCI", "KOR", 3, "Etihad Airways")
p3 = Passenger("P3", "P", "F", "Ben A", 42587961558, "F1340", "MCI", "CHI", 3, "Delta Airlines")

#Instances of Employee class
e1 = Employee("E1", "E", "M", "Greg C", 142587961558, "F1578", "EWR", "LHR", 3, "Etihad Airways")
e2 = Employee("E2", "E", "M", "John O", 422587961558, "F1234", "EWR", "LHR", 3, "Etihad Airways")
e3 = Employee("E2", "E", "M", "Dale W", 424587961558, "F1647", "EWR", "ABC", 3, "Delta Airlines")

#This method prints the travel details for the passenger
p1.get_travel_details_passanger()
#This method prints the travel details for the employee
e1.get_travel_details_employee()

#This method prints the travelling passengers on that Flight
p1.get_travelling_passengers()

#Prints the boarding pass
T1 = Ticket("P1", "P", "M", "Cory", 4258796358, "EY2335", "MCI", "KOR", 3, "Etihad Airways", "G", "E", 12)
T1.get_boarding_pass("Cory")