# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 4a: Predicting the Spread of West Nile Virus

## Problem Statement
Team **39 SIR** of the  **D**isease **A**nd **T**reatment **A**gency, division of **S**ocietal **C**ures **I**n **E**pidemiology and **N**ew **C**reative **E**ngineering (DATA-SCIENCE) is to effectively plan the deployment of pesticides in order to mitigate the spread of the West Nile Virus in Chicago City.

This will be done by analysing the data given by the Department of Public Health to produce the following deliverables:

1. A predictive model to facilitate informed decision making by the city of Chicago when it decides where to spray the pesticides.

2. Cost-Benefit Analysis of the annual cost projections for various levels of pesticide coverage (cost) and the effect of these various levels of pesticide coverage (benefit).

## Background
West Nile virus (WNV) is the leading cause of mosquito-borne disease in the continental United States. 

It is most commonly spread to people by the bite of an infected mosquito. Cases of WNV occur during mosquito season, which starts in the summer and continues through fall.

Most people infected with WNV do not feel sick:
* About 1 in 5 people who are infected develop a fever and other symptoms
* About 1 out of 150 infected people develop a serious, sometimes fatal, illness.

**Source**: https://www.cdc.gov/westnile/index.html

## Datasets
There are 4 datasets sponsored by the CDC: `train`, `test`, `weather` and `spray`.

### Data Dictionary

`train_df`

Period: 2007, 2009, 2011, and 2013

|Feature|Type|Description|
|:---|:---:|:---|
|<b>Date</b>|*object*|Date that the WNV test is performed|
|<b>Address</b>|*object*|Approximate address of the location of trap, this is used to send to the GeoCoder|
|<b>Species</b>|*object*|The species of mosquitos|
|<b>Block</b>| *int64*|Block number of address|
|<b>Street</b>|*object*|Street name|
|<b>Trap</b>|*object*|Id of the trap|
|<b>AddressNumberAndStreet</b>|*object*|Approximate address returned from GeoCoder|
|<b>Latitude</b>|*float64*|Latitude returned from GeoCoder|
|<b>Longitude</b>|*float64*|Longitude returned from GeoCoder|
|<b>AddressAccuracy</b>|*int64*|Accuracy returned from GeoCoder|
|<b>NumMosquitos</b>|*int64*|Number of mosquitoes caught in this trap|
|<b>WnvPresent</b>|*int64*|Whether West Nile Virus was present in these mosquitos. 1 means WNV is present, and 0 means not present. |

<br>

`test_df`

Period: 2008, 2010, 2012, and 2014

|Feature|Type|Description|
|:---|:---:|:---|
|<b>Id</b>|*int64*|The id of the record|
|<b>Date</b>|*object*|Date that the WNV test is performed|
|<b>Address</b>|*object*|Approximate address of the location of trap, this is used to send to the GeoCoder|
|<b>Species</b>|*object*|The species of mosquitos|
|<b>Block</b>| *int64*|Block number of address|
|<b>Street</b>|*object*|Street name|
|<b>Trap</b>|*object*|Id of the trap|
|<b>AddressNumberAndStreet</b>|*object*|Approximate address returned from GeoCoder|
|<b>Latitude</b>|*float64*|Latitude returned from GeoCoder|
|<b>Longitude</b>|*float64*|Longitude returned from GeoCoder|
|<b>AddressAccuracy</b>|*int64*|Accuracy returned from GeoCoder|

<br>

`weather_df`

Period: 2007, 2008, 2009, 2010, 2011, 2012, 2013, and 2014

|Feature|Type|Description|
|:---|:---:|:---|
|<b>Date</b>|*object*|Date of record|
|<b>Station</b>|*int64*|Station number, either 1 or 2|
|<b>Tmax</b>|*int64*|Maximum temperature in Degrees Fahrenheit|
|<b>Tmin</b>|*int64*|Minimum temperature in Degrees Fahrenheit|
|<b>Tavg</b>|*object*|Average temperature in Degrees Fahrenheit|
|<b>Depart</b>| *object*|Temperature departure from normal in Degrees Fahrenheit|
|<b>DewPoint</b>|*int64*|Average Dew Point in Degrees Fahrenheit|
|<b>WetBulb</b>|*object*|Average Wet Bulb in Degrees Fahrenheit|
|<b>Heat</b>|*object*|Absolute temperature difference of Tavg from base temperature of 65 Degrees Fahrenheit if Tavg < 65|
|<b>Cool</b>|*object*|Absolute temperature difference of Tavg from base temperature of 65 Degrees Fahrenheit if Tavg > 65|
|<b>Sunrise</b>|*object*|Time of Sunrise (Calculated, not observed)|
|<b>Sunset</b>|*object*|Time of Sunset (Calculated, not observed)|
|<b>CodeSum</b>|*object*|Weather Phenomena, refer to CodeSum Legend below|
|<b>Depth</b>|*object*|Snow / ice in inches|
|<b>Water1</b>|*object*|Water equivalent of Depth|
|<b>SnowFall</b>| *object*|Snowfall in inches and tenths|
|<b>PrecipTotal</b>|*object*|Rainfall and melted snow in inches and hundredths|
|<b>StnPressure</b>|*object*|Average station pressure in inches of HG|
|<b>SeaLevel</b>|*object*|Average sea level pressure in inches of HG|
|<b>ResultSpeed</b>|*float64*|Resultant wind speed in miles per hour|
|<b>ResultDir</b>|*int64*|Resultant wind direction in Degrees|
|<b>AvgSpeed</b>|*object*|Average wind speed in miles per hour|

<br>

`spray_df`

Period: 2011, and 2013

|Feature|Type|Description|
|:---|:---:|:---|
|<b>Date</b>|*object*|Date of the spray|
|<b>Time</b>|*object*|Time of the spray|
|<b>Latitude</b>|*float64*|Latitude returned from GeoCoder|
|<b>Longitude</b>|*float64*|Longitude returned from GeoCoder|

<br>

## Recommendations

### WHEN
to spray, based on the total number of mosquitoes trapped monthly.

**Monitoring: Traps** Continue from May through Oct

**Action: Spray** When traps hit 14% WNV-positive

### WHERE
to spray, based on the top 10 hot spots.

1. ORD Terminal 5, O'Hare International Airport
2. South Doty Avenue
3. 4100 North Oak Park Avenue
4. South Stony Island Avenue
5. 4600 Milwaukee Avenue
6. 8200 South Kostner Avenue
7. 2400 East 105th Street
8. 3600 North Pittsburgh Avenue
9. O’Hare Court, Bensenville
10. 7000 North Moselle Avenue

## Future Research

- Analyse effect of birds on WNV infection
    - Birds are amplifying hosts (Environmental Research and Public Health, 2020)

- Analyse the severity of WNV cases 
    - Look at total no. of cases instead of binary outcomes

