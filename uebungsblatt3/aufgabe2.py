
print("Aufgabe 2")

x = raw_input("please enter C or F to converte your tempreature: ")

if (x == "f")or (x == "F"):
    fahrenheit = input("Fahrenheit: ")

    if not(fahrenheit < -459.67):
        celsius = (fahrenheit - 32) / 1.8
        print "celsius: ", celsius
        
    else:
        print "the absolute zero of fahrenheit: -459.67 F"


elif (x == "c")or (x == "C"):
    celsius = input("celsius: ")

    if not(celsius < -273.15):
        fahrenheit = (celsius * 1.8) + 32
        print "Fahrenheit: ", fahrenheit

    else:
        print "the absolute zero of celsius: -273.15 C"

elif (x == "c")or (x == "C") or (x == "f")or (x == "F"):
    print"err restart plese"

print "ende"
