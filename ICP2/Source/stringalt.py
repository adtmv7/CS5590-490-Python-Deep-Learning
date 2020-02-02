#string_alternative and call it from main function

String = input("Enter String =")            #this is input string
def string_alternative():
    Output = ""
    for i in range(len(String)):
        if (i%2==0):                        #even index checking
            Output = Output+String[i]
    print(Output)
string_alternative()