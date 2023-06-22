#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing modules/libraries
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings


# In[2]:


warnings.filterwarnings("ignore")


# In[20]:


zomato = pd.read_csv("C:/Users/saksh/OneDrive - CHRIST UNIVERSITY/christ/sem 3/Data Lab/Zomato dataset.csv",encoding='latin1')
zomato.head(2)


# In[ ]:


zomato.head(15)


# In[ ]:


print("Total rows in the dataset: " , len(zomato.axes[0]))
print("Total columns in the dataset: ", len(zomato.axes[1]))


# In[ ]:


df1 = pd.read_excel("C:/Users/saksh/OneDrive/Documents/christ/sem 3/Data Lab/Country-Code.xlsx")
df1.head()


# In[ ]:


zomato_country = pd.merge(zomato,df1,on='Country Code',how='left')
zomato_country.head(2)


# In[ ]:


zomato_country.rename(columns = {'f':"Restaurant ID"}, inplace = True)
zomato_country.head()


# In[ ]:


zomato_india = zomato_country[zomato_country['Country Code'] == 1]
zomato_india.head()


# In[ ]:


zomato_country[["Average Cost for two"]]


# In[ ]:


zomato_india.head(15)


# In[ ]:


## Checking if dataset contains any null

nan_values = zomato.isna()
nan_columns = nan_values.any()

columns_with_nan = zomato.columns[nan_columns].tolist()
print(columns_with_nan)


# In[ ]:


zomato_india


# In[ ]:


print('List of counteris the survey is spread accross - ')
for x in pd.unique(zomato_country.Country): print(x)
print()
print('Total number to country', len(pd.unique(zomato_country.Country)))


# In[ ]:


ratings = zomato_country.groupby(['Aggregate rating','Rating color', 'Rating text']).size().reset_index().rename(columns={0:'Rating Count'})
ratings


# In[ ]:


No_rating = zomato_country[zomato_country['Rating color']=='White'].groupby('Country').size().reset_index().rename(columns={0:'Rating Count'})
No_rating


# In[ ]:


zomato_india['City'].value_counts()


# In[ ]:


# Import the required library
from geopy.geocoders import Nominatim

# Initialize Nominatim API
geolocator = Nominatim(user_agent="MyApp")
a= input("Enter the location = ")
location = geolocator.geocode(a)

latitude=location.latitude
longitude=location.longitude
print(latitude, longitude)


# In[ ]:


zomato_india["Restaurant Name"].unique()


# In[ ]:


zomato_india["Cuisines"].unique()


# In[ ]:


input_dataset = pd.DataFrame()
def dataset_input(a):
    zomato_india[(zomato_india["Locality Verbose"].str.contains(a, case=False)|(zomato_india["Locality"].str.contains(a, case=False)))]
    input_dataset = zomato_india[(zomato_india["Locality"].str.contains(a, case=False))|(zomato_india["Locality Verbose"].str.contains(a, case=False))]
    return input_dataset
    


# In[ ]:


input_location= input("Enter the location: ")
dataset_input(input_location)


# In[ ]:





# In[ ]:


zomato_ncr = zomato_india[(zomato_india['City'] == 'New Delhi') | (zomato_india['City'] == 'Gurgaon') | (zomato_india['City'] == 'Faridabad') | (zomato_india['City'] == 'Noida')]
zomato_ncr.head()


# In[ ]:


zomato_country.groupby(['Country','Has Online delivery']).size().reset_index()


# In[ ]:


zomato_country[zomato_country['Has Online delivery']=='Yes'].Country.value_counts()


# In[ ]:


zomato_country.groupby(['Has Online delivery']).size().reset_index()


# In[ ]:


zomato_india.groupby(['Has Online delivery']).size().reset_index()


# In[ ]:


zomato_ncr.shape 


# In[ ]:


zomato_india["Average Cost for two"]


# In[ ]:


zomato_ncr.info()


# In[ ]:


zomato_ncr.describe()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
encoder= LabelEncoder()


# In[ ]:


zomato_india['City_en']=encoder.fit_transform(zomato_india['City'])


# In[ ]:


zomato_india['City_en'].value_counts()


# In[ ]:


tableBooking = LabelEncoder()
tableBooking.fit(zomato_india['Has Table booking'])                    


# In[ ]:


onlineBooking= LabelEncoder()

onlineBooking.fit(zomato_india['Has Online delivery'])


# In[ ]:


rating_text= LabelEncoder()
rating_text.fit(zomato_india["Rating text"])


# In[ ]:


zomato_india['Online Booking'] = onlineBooking.transform(zomato_india['Has Online delivery'])


# In[ ]:


zomato_india['Table Booking'] = tableBooking.transform(zomato_india['Has Table booking'])


# In[ ]:


zomato_india['Rating text']=encoder.fit_transform(zomato_india['Rating text'])


# In[ ]:


zomato_india.head()


# In[ ]:


encoded_data=pd.get_dummies(zomato_india, columns=['City_en','Rating text'],drop_first=True)
encoded_data.head()


# #get rid of price range because its basically another ouput
# #get rid of 'is delivering now' because lots of bias

# In[ ]:


encoded_data.drop(columns=['Restaurant ID','Restaurant Name','Is delivering now','Switch to order menu','Price range','Rating color'],axis=1,inplace=True)


# In[ ]:


encoded_data.drop(columns=['Address','Locality','Locality Verbose'],axis=1,inplace=True)


# In[ ]:


encoded_data.drop(columns=['Country Code','Currency',"Country"],axis=1,inplace=True)


# In[ ]:


encoded_data.drop(columns=['Has Table booking','Has Online delivery',"City"],axis=1,inplace=True)


# In[ ]:


encoded_data.rename({"City_en":"City"},inplace=True)


# In[ ]:


encoded_data.head()


# In[ ]:


encoded_data.groupby('Cuisines').mean().head()


# In[ ]:


cuisines=encoded_data.groupby('Cuisines').mean()['Average Cost for two'].reset_index()
cuisines


# In[ ]:


cuisines.shape


# In[ ]:


#merging cuisine with earlier dataset df

encoded_data=encoded_data.merge(cuisines,on='Cuisines')
encoded_data.head()


# In[ ]:


encoded_data.drop(columns=['Cuisines'],axis=1,inplace=True)


# In[ ]:


encoded_data.rename(columns={'Average Cost for two_y':'Cuisines'},inplace=True)
encoded_data.head()


# In[ ]:


#1:Extract X and Y

X=encoded_data.drop(columns=['Average Cost for two_x']).values
Y= encoded_data["Average Cost for two_x"]


# In[ ]:


X


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg=LinearRegression()


# In[ ]:


reg.fit(X_train,Y_train)


# In[ ]:


Y=encoded_data['Average Cost for two_x'].values


# #algo giving predicted prices for all the resturaunts 

# In[ ]:


reg.predict(X_test)


# In[ ]:


#now storing the pred values in variable  Y_pred
Y_pred=reg.predict(X_test)


# In[ ]:


print (X_test.shape)
print(Y_pred.shape)
print (Y_test.shape)


# In[ ]:


#now we have to compare Y_test with Y_pred to determine efficacy of the model
Y_pred[2]


# In[ ]:


Y_test


# In[ ]:


#Now we find r2 score to determine how well model is working
from sklearn.metrics import r2_score
r2_score(Y_test,Y_pred)
# it is found that the regression is around 70% accurate


# In[ ]:


# Encode the boolean values
boolean_columns = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
# Create encoding
encoding = {"Yes" : 1,
             "No" : 0}
# encoding using the lambda function
for col in boolean_columns:
    zomato_ncr[col] = zomato_ncr[col].apply(lambda x : encoding[x])


# In[ ]:


zomato_ncr[boolean_columns]


# In[ ]:


# check restaurants having more than 7 outlets in ncr
more_than_7 = {}
for restaurants,val in zip(zomato_ncr['Restaurant Name'].value_counts().index, zomato_ncr['Restaurant Name'].value_counts()):
    if val >= 7:
        more_than_7[restaurants] = val
print("Reataurants with more than 7 outlets in ncr: {}".format(len(more_than_7.keys())))


# In[ ]:


# check restaurants having more than 7 outlets in india
more_than_7_ = {}
for restaurants,val in zip(zomato_india['Restaurant Name'].value_counts().index, zomato_india['Restaurant Name'].value_counts()):
    if val >= 7:
        more_than_7_[restaurants] = val
print("Reataurants with more than 7 outlets in India: {}".format(len(more_than_7_.keys())))


# In[ ]:


pd.set_option('display.max_rows', None)
zomato_ncr.head(3)


# In[ ]:


zomato_ncr.drop(['Country Code','Currency'], axis = 1, inplace = True)  # dropping unnecessary columns what are not needed


# In[ ]:


zomato_ncr.head()


# In[ ]:


# Visualisation of top 5 restaurants in ncr
zomato_ncr['Restaurant Name'].value_counts().head(5).plot(kind='bar', color='black', figsize=(10,10))
plt.xlabel('Name of restaurant', color='g', fontsize = 18)
plt.ylabel('Amount of each restaurant', color='g', fontsize = 18)
plt.title('Top 5 Restaurants with most outlets in Delhi NCR', color = 'g', fontsize = 25)
plt.figure();


# # Inference:
# Delhi had the most data in my dataset. So according to the locality I combined all the city of delhi to make delhi_ncr's data and then check which retaurants has the most number of outlets or more liked. If a particular restaurant chain is increasing even after a desired average that means people are liking the food of that restaurant so using the count function I tried to find out the most liked restaurants in that area.

# In[ ]:


plt.figure(figsize = (22, 11))
plt.bar(x = more_than_7.keys(), height = more_than_7.values(), color='royalblue')
plt.xlabel('Restaurants', fontsize = 18)
plt.ylabel('Number of outlets', fontsize = 19)
plt.title('Restaurants having more than 7 outlets in Delhi NCR', fontsize = 25)
plt.xticks(rotation = -90)
plt.yticks(np.arange(0,90,2))
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.show();


# # inference:
# this shows all the restaurants in delhi ncr that has more than 7 outlets

# In[ ]:


# Visualisation of top 5 restaurants in ncr
zomato_india['Restaurant Name'].value_counts().head(5).plot(kind='bar', color='black', figsize=(10,10))
plt.xlabel('Name of restaurant', color='g', fontsize = 18)
plt.ylabel('Amount of each restaurant', color='g', fontsize = 18)
plt.title('Top 5 Restaurants with most outlets in India', color = 'g', fontsize = 25)
plt.figure();


# # inference:
# this shows the most outlet restaurants in india. according to the graph above and this one we can clearly see delhi has most of them and rest others are in different city

# In[ ]:


plt.figure(figsize = (22, 11))
plt.bar(x = more_than_7_.keys(), height = more_than_7_.values(), color='royalblue')
plt.xlabel('Restaurants', fontsize = 18)
plt.ylabel('Number of outlets', fontsize = 19)
plt.title('Restaurants having more than 7 outlets', fontsize = 25)
plt.xticks(rotation = -90)
plt.yticks(np.arange(0,90,2))
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.show();


# # inference:
#  this shows restaurants in India that has more than 7 outlets. which again shows very slight changes compare to the graph that came in delhi ncr report
#  

# In[ ]:


def seven_or_more(data, column):
    for col in column:

        d = zomato_ncr[zomato_ncr['Restaurant Name'] == col]
        data = pd.concat((data,d), axis = 0)
  
    return data


# In[ ]:


seven_or_more_outlets = pd.DataFrame(None)
seven_or_more_outlets = seven_or_more(seven_or_more_outlets, list(more_than_7.keys()))
seven_or_more_outlets.shape


# In[ ]:


seven_or_more_outlets.head()


# In[ ]:


plt.figure(figsize = (12,10))
sb.countplot(seven_or_more_outlets['City'], palette="Set3")
plt.yticks(np.arange(0,950,50))
plt.show()


# In[ ]:


# analysis on new delhi data
delhi_data = seven_or_more_outlets.groupby('City').get_group('New Delhi')
delhi_data.head()


# In[ ]:


delhi_data[['Locality','Locality Verbose']].head()


# In[ ]:


delhi_data.drop('Locality Verbose',axis = 1, inplace = True)


# In[ ]:


delhi_data['Locality'].value_counts().sort_values(ascending=False).head(10)


# In[ ]:


plt.figure(figsize = (12, 30))
plt.barh(delhi_data['Locality'].value_counts().sort_values().index, delhi_data['Locality'].value_counts().sort_values(), color = 'royalblue')
plt.ylabel('New Delhi Localities', fontsize = 18)
plt.xlabel('Count', fontsize = 18)
plt.title('Distribution of Restaurants in New Delhi Localities', fontsize = 30)
plt.xticks(np.arange(0,26,1))
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.show()


# In[ ]:


plt.figure(figsize=(15,23))
plt.barh(delhi_data['Cuisines'].value_counts().sort_values().index, delhi_data['Cuisines'].value_counts().sort_values(), color = 'royalblue')
plt.ylabel("Cuisines", fontsize = 30)
plt.xlabel("Count", fontsize = 30)
plt.title('Visualising Popularity of various cuisines in Delhi', fontsize = 40)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.show()


# In[ ]:


# Creating competitor data 
competitor = set()
for res,cuisine in zip(delhi_data['Restaurant Name'], delhi_data['Cuisines']):
    if 'Mughlai' in cuisine or 'Fast Food' in cuisine or 'American' in cuisine or 'Pizza' in cuisine or 'Burger' in cuisine or 'Biryani' in cuisine:
        competitor.add(res)
competitor


# In[ ]:


sb.countplot(zomato_country['City'])
sb.countplot(zomato_country['City']).set_xticklabels(sb.countplot(zomato_country['City']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(30,13)
plt.title('Location wise count for restaurants')


# # inference:
# this shows count of restaurants in each city across the globe

# In[ ]:


sb.countplot(zomato_india['City'])
sb.countplot(zomato_india['City']).set_xticklabels(sb.countplot(zomato_india['City']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf();
fig.set_size_inches(12,13)
plt.title('Location wise count for restaurants')


# this graph shows how many restaurants exists in each city or locality

# In[ ]:


a=pd.merge(zomato_country[(zomato_country['Country']=='India')].groupby(['City']).agg({'Restaurant ID':np.count_nonzero}).reset_index(),zomato_country[(zomato_country['Has Online delivery']=='Yes')&(zomato_country['Country']=='India')].groupby(['City']).agg({'Restaurant ID':np.count_nonzero}).reset_index(),on='City',how='inner')                                                    
a.rename(columns={'Restaurant ID_x':'Total_restaurants','Restaurant ID_y':'Online_restaurants'},inplace=True)
a['%age']=a['Online_restaurants']*100/a['Total_restaurants']
a['%age']=a['%age'].apply(lambda x: np.round(x,2))
a.sort_values('Online_restaurants',ascending=False).reset_index(drop=True)


# In[ ]:


#Encode the input Variables
def Encode(zomato):
    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:
        zomato[column] = zomato[column].factorize()[0]
    return zomato

zomato_en = Encode(zomato.copy())
zomato_en.head() # looking at the dataset after transformation


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


#Defining the independent variables and dependent variables
x = zomato_en.iloc[:,[2,3,5,6,7,8,9,11]]
y = zomato_en['Aggregate rating']
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)
x_train.head()


# In[ ]:


#Create a pie chart for cities distribution
city_values=zomato_india.City.value_counts().values
city_labels=zomato_india.City.value_counts().index


# In[ ]:


plt.pie(city_values[:5],labels=city_labels[:5],autopct='%1.2f%%')
plt.show()


# In[ ]:


import plotly.graph_objects as go

labels = list(zomato_india.City.value_counts().index)
values = list(zomato_india.value_counts().values)

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.6,title="Zomato's Presence Citywise")])
fig.show()



# In[ ]:


#Count plot
sb.countplot(x="Rating color", data=ratings,palette=['white','red','orange','yellow','cyan','green']);


# # inference:
# this graph shows the text rating color code. where 1 person has marked it white which means the review was very poor. 7 has given red which means those texts has poor ratings, 10 gave orange that means they found the food okay. 5 has marked the restaurants yellow, green and dark green accordingly that means good, very good and excellent.

# In[ ]:


# Rating Text Countplot
import matplotlib
matplotlib.rcParams['figure.figsize'] = (18,6)

sb.barplot(x="Rating text",y="Rating Count",data=ratings)
plt.xlabel("Rating Text",{"color":"red"},size=18)
plt.ylabel("Count",{"color":"red"},size=18)
plt.title("Rating Text Count",size=30,color='red')
plt.show()
plt.savefig("rating text count.jpg")


# # Inference:
# this graph shows how more than 2000 restaurants has not been rated and less than 50 has been rated poor, around 350-4oo has been rated average and around 400-450 are rated good. less than 200 restaurants  are rated very good and hardly 50 has been rated excellent. 

# In[ ]:


# Ratings Countplot with Colour Mapping
sb.barplot(x="Aggregate rating",y="Rating Count",data=ratings,hue="Rating color",palette=['white','red','orange','yellow','cyan','green'])
plt.xlabel("Ratings",{"color":"red"},size=18)
plt.ylabel("Count",{"color":"red"},size=18)
plt.title("Ratings Count",size=30,color='red')
plt.show()
plt.savefig("ratings count.jpg")


# # Inference:
# this graph shows how different restaurants has been rated with different colors. this has the count of different ratings starting from white to green. 
# 1. from 0.0-2.0: white
# 1. 2.1-2.4: red
# 1. 2.5-3.4: orange
# 1. 3.5-3.9: yellow
# 1. 4.0-4.5: green
# 1. 4.5-5: dark green

# In[ ]:


# Let's Analyze which City has Maximum Price Range
highPrice_city = zomato_country.groupby(['Country','City']).mean()['Price range'].reset_index().sort_values("Price range",ascending=False)[:10]
highPrice_city.style.set_properties(**{'color': 'orange'})


# In[ ]:


# Let's Analyze which City has Minimum Price Range
lowPrice_city = zomato_country.groupby(['Country','City']).mean()['Price range'].reset_index().sort_values("Price range",ascending=False)[-10:]
lowPrice_city.style.set_properties(**{'color': 'green'})


# In[ ]:


# Let's see top 10 Cuisines
zomato_country.groupby('Cuisines').size().reset_index().sort_values(0,ascending=False).rename(columns={0:'Count'})[:10]


# In[ ]:


# Let's check that Cities which are currently delivering orders
zomato_india[zomato_india['Is delivering now']=='Yes'].groupby(["Country","City"]).count()["Is delivering now"].reset_index()


# In[ ]:


cuisine_val = zomato_india.Cuisines.value_counts()             
cuisine_label = zomato_india.Cuisines.value_counts().index 
fig = go.Figure(data=[go.Pie(labels=cuisine_label[:10], values=cuisine_val[:10], textinfo='label+percent',
                             insidetextorientation='radial',title='Top 10 Cuisine'
                            )])
fig.show();


# In[ ]:


from pprint import pprint

cuisines_total = zomato_country['Cuisines']
texts = []
hash_map = set()
cuisines_total.dropna(inplace=True)
for i in cuisines_total:
    for j in str(i).split(', '):
        if j not in hash_map:
            texts.append(j.lower())
            hash_map.add(j)
pprint(texts)


# In[ ]:


temp = ratings[['Rating text', 'Rating Count']].groupby(['Rating text']).sum()
temp.drop('Not rated', axis=0, inplace=True)


# In[ ]:


plt.pie(temp['Rating Count'], labels = ['Average', 'Excellent', 'Good', 'Poor', 'Very Good'], autopct='%1.2f')


# The range of ratings for categories are as follows.
# 1.8 - 2.4: poor
# 2.5 - 3.4: Average
# 3.5 - 3.9: Good
# 4.0 - 4.4: Very Good
# 4.4 - 4.9: Excellent
# The mean rating is 3.24 and 95% of the values lie between 2.15 and 4.34.
# 
# Due to low number of poor ratings on restaurants it can be concluded that people are liking the restaurants and are rating according to the actual experience
# 

# In[ ]:


zomato_india.groupby('City')['Average Cost for two'].mean().sort_values(ascending =False)[-3:]


# In[ ]:


zomato_country[["Currency", "Country"]].groupby("Country").first()


# In[ ]:


zomato_country.Country.value_counts()  


# In[ ]:


zomato_india.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False).head(50).plot(kind='bar', color = 'orange', figsize=(20,5), title='TOP 50 cuisines')


# # Inference:
# this shows famous cuisines according to the rating across India. 

# In[ ]:


zomato_india.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False).tail(50).plot(kind='bar',figsize=(20,5), title='Rating the worst 50 cuisines')


# # INFERENCE:
# this cuisines and thecombo is not liked at all by the people of India or people just didn't take gtime out to rate the cuisine

# In[ ]:


import plotly.graph_objects as go
import plotly.express as px


# In[ ]:


def locality_dist(cityname):
    plt.figure(figsize = (8,25))
    plt.barh(zomato_india[zomato_india['City'].str.lower()==cityname.lower()]['Locality'].value_counts().sort_values().index,
             zomato_india[zomato_india['City'].str.lower()==cityname.lower()]['Locality'].value_counts().sort_values())
    plt.ylabel('{} Localities'.format(cityname), fontsize = 15)
    plt.xlabel('Count', fontsize = 15)
    plt.title('Distribution of Restaurants in {} Localities'.format(cityname), fontsize = 20)
    plt.grid( linestyle='-', linewidth=1, axis='x', alpha=0.5)


# In[ ]:


def Locality_count_graph(cityname):
    city_name= zomato_india[(zomato_india.City.str.contains(cityname,case=False))]
    fig = go.Figure(data=[go.Bar(
    x=city_name.Locality.value_counts().head(10).index,
    y=city_name.Locality.value_counts().head(10))])
    
    fig.update_xaxes(title_text="Locality in " + city)
    fig.update_yaxes(title_text="Count")


    return fig.show();


# In[ ]:


def has_online_delivery(cityname):
    locality_name= zomato_india[(zomato_india.City.str.contains(cityname,case=False))]
    top_locality = locality_name.Locality.value_counts().head(10)
    plt.figure(figsize=(12,6))
    ax = sb.countplot(y= "Locality", hue="Has Online delivery", data=locality_name[locality_name.Locality.isin
                                                                                   (top_locality.index)])
    plt.title('Resturants Online Delivery');


# In[ ]:


def has_table_booking(cityname):
    locality_names= zomato_india[(zomato_india.City.str.contains(cityname,case=False))]
    fig = px.bar(locality_names, 
                   x=locality_names.Locality, 
                   #marginal='box', 
                   color=locality_names['Has Table booking'],
             #y=k.Locality.value_counts().head(10).index, 
                   #color_discrete_sequence=['Grey', 'Red','Orange','Yellow','Green',"Darkgreen"], 
                   title='Table Booking options')
    fig.update_layout(bargap=0.1)
    return fig.show();


# In[ ]:


def costly_restaurants(cityname):
    locality_name_= zomato_india[(zomato_india.City.str.contains(cityname,case=False))]
    expensive_rest=locality_name_.nlargest(25, 'Average Cost for two')
    plot = px.bar(expensive_rest, x='Restaurant Name', 
            y='Average Cost for two',
            hover_data=['Has Online delivery'] ,
            title = 'Top 25 costly restaurant')
    return plot.show();


# In[ ]:


def valid_to_open(cityname):
    locality_names_= zomato_india[(zomato_india.City.str.contains(city,case=False))]
    g = sb.catplot(data=locality_names_ ,x= "Restaurant Name", y= "Average Cost for two", hue= "Aggregate rating",
                   height = 5, aspect = 5)
    g.set_xticklabels(rotation=30)
    return plt.show()


# In[ ]:


def review_ratings(cityname):
    locality_names= zomato_india[(zomato_india.City.str.contains(cityname,case=False))]
    fig = px.bar(locality_names, 
                   x=locality_names.Locality, 
                   #marginal='box', 
                   color=locality_names['Rating color'],
             #y=k.Locality.value_counts().head(10).index, 
                   #color_discrete_sequence=['Grey', 'Red','Orange','Yellow','Green',"Darkgreen"], 
                   title='Reviews Ratings')
    fig.update_layout(bargap=0.1)
    return fig.show();


# In[ ]:


city= input("Enter the name of the city: ")


# In[ ]:


review_ratings(city)


# In[ ]:


valid_to_open(city)


# In[ ]:


locality_dist(city)


# In[ ]:


has_online_delivery(city)


# In[ ]:


has_table_booking(city)


# In[ ]:


costly_restaurants(city)


# # Inference:
# This five graph together helps a person to figure out whether they should be opening a restaurant in that particular locality or not and what all can they do after analyzing the data and condition of other restaurants that already exists in there. 
# an input is asked by the user which is name of the city. once the user inputs that they get following information
# 
# 1. Review Rating- X axis has all the localities of that city according to the data and Y axis has count of number of restaurant in that area according to the data according to the review color. the seequence of rating is 'Grey', 'Red','Orange','Yellow','Green',"Darkgreen". Velachery has 2 restaurants with dark green rating.
# 
# 1. locality distribution - in this we can see which locality has most number of restaurants. So if a locality has more than 10 restaurants and according to the rating chart we can figure out whether that restaurant is loved or not and then we can check with their cuisine and someone who wants to open a restaurant in that area can keep all these quantifiers in their head before starting something new. Velachery ha sthe most restaurants. 
# 
# 1. has table booking option- this graph shows us whether the restaurants that are there have table booking option or not. so if you want to pre-book your table and other restaurants in that area doesn't have that feature, your outlet can have that and it will make it stand out possibly in that area.
# 
# 1. has online delivery - this graph shows whether the restaurants in that locality has online delivery option or not. we can see in velachery none of the restaurant has online delivery option.
# 
# 1. valid to open- this shows each restaurants data and their aggregate rating in comparison to the aggregate price for two. so, if a particular restaurant has a aggregate rating of 3.8 and the cost for two is 800 which means it's food is moderately liked by the people in there.  we can see ab's barbeques restaurant has an average rating of 4.9 and the range price for two is between 1500-1750 but people are still enjoying. This means if a restaurant with same kind of cuisine is opened in that area at a chepar price range, the probability of it working is likely.
# 1. costly restaurants in that area- if according to the above data we figure out some stats about a particular data, the next task is to figure out whether that is the costliest restaurant across the city or just in that area. this will help a person to set a mark for their business before starting something new.
# 

# In[ ]:


cuisine_data = zomato_india.groupby(['Cuisines'], as_index=False)['Restaurant ID'].count()
cuisine_data.columns = ['Cuisines','Number of Resturants']
Top15= (cuisine_data.sort_values(['Number of Resturants'],ascending=False)).head(15)
sb.set(rc={'figure.figsize':(11.7,8.27)})
sb.barplot(Top15['Cuisines'], Top15['Number of Resturants'])
plt.xlabel('Cuisines', fontsize=20)
plt.ylabel('Number of Resturants', fontsize=20)
plt.title('Top 15 cuisines on Zomato', fontsize=20)
plt.xticks(rotation = 90)
plt.show()


# # INFERENCE:
# this graph shows the top 15 cuisines across India according to the data. The bar chart representation is in form of counts of each restaurant done using groupby feature. We can see North Indian food is the most spread across and liked. Most of the restaurants has north Indian food followed with north indian and chinese and then fast food and according to the least spread across it is bakery aong with fast food.

# In[ ]:


zomato_india['Restaurant Name'].value_counts().head(25).plot(kind='pie',figsize=(13,13), title="Top 25 Restaurants with maximum outlets", autopct='%1.1f%%');
plt.savefig("25 restaurants with maximum outlets.jpg")


# # INFERENCE:
# this graph shows the top 15 restaurant across India according to the data. The pie chart representation is in form of the most number of the outlet of each restaurant across the data and India. We can see CCD has the most outlet followed with dominos and subway and according to the least spread across restaurant is haldiram's and burger king

# In[ ]:


zomato_country.to_csv('cleaned miniproject dataset 21112022.csv')

