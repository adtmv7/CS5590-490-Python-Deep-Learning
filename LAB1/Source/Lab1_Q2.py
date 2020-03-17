#2) Concatenate two dictionaries and sort the concatenated dictionary by value.

dictone={'Class':24,'Name':123} # create first dictionary
dicttwo={'Python':490,'UMKC':2020} # create second dictionary
mergeddict = {**dictone,**dicttwo} # merge contents of 2nd dictionary in first
print("Concatenated Dictionary is:")
print(mergeddict)
print(sorted(mergeddict))
print(sorted(mergeddict.values())) # sorting the merged dictionary by values

