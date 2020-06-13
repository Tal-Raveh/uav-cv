import pickle

with open('D:/Documents/אקדמיה/תואר שני - הנדסת מכונות/תיזה/workspace/result.loc' , "rb") as file:
    location = pickle.load(file)

print(location)