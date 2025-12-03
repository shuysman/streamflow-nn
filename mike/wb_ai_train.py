#!/home/ubuntu/anaconda3/bin/python

#print ("Content-Type: text/html\n")

import os,sys
import warnings
warnings.filterwarnings("ignore")
os.environ['TMPDIR']='./'
import cgi,sys,datetime,read_daily_data,numpy
from scipy import stats
from calendar import monthrange
import numpy
import matplotlib
import matplotlib.style
import matplotlib.patches as patches
from io import BytesIO
import base64
#print("Content-Type: text/html\n")
web = True
if web == True:
    daily_file_path = '../daily/'
    monthly_file_path = '../monthly/'
    station_info_path = '../'
    matplotlib.use('Agg')
    
else:
    station_info_path = ''
    daily_file_path = ''

import matplotlib.pyplot as plt
    
pi = numpy.pi
e = numpy.e
sys.stderr = sys.stdout
missing_val_list = [9999,99999,-99999,'9999','-99999','-9999',-9999]

class autodict():
    def __init__(self,paramchecklist):
        self.d = {}
        self.paramchecklist = paramchecklist
    def __getitem__(self,key):
        return self.d[key]
    def __setitem__(self,key,val):
        self.d[key] = val
    def rectify_two_keys(self,key1,key2):
        mask = numpy.isnan(self.d[key1]) == True
        self.d[key2][mask] = numpy.nan
    def add(self,key,item):
        if key not in self.d:
            self.d[key] = [item]
        else: 
            if type(self.d[key]) == numpy.ndarray: 
                numpy.append(self.d[key],item)
            else:
                self.d[key].append(item)
    def make_arrays(self): # Appending to lists is faster than to np arrays so don't call this until all data is read.
        for key in self.d: self.d[key] = numpy.array(self.d[key])
    def raise_missing_errors(self,cat_key='year'):
        catlist = list(set(self.d[cat_key]))
        for cat in catlist:
            this_mask = self.d[cat_key] == cat
            for key in self.paramchecklist:
                if numpy.all(numpy.isnan(self.d[key][this_mask])) == True: 
                    raise Exception('One of the necessary parameters is entirely missing for an entire year.')
            
class soil_reservoir():
    def __init__(self,max_holding_capacity):
        self.stored_water = 0.0
        self.max = float(max_holding_capacity)
    def add(self,amount):
        if numpy.isnan(amount) == False:# Keep the old soil water value if missing data. Number of missing values is tracked elsewhere.
            self.stored_water = self.stored_water + amount
        runoff = nonneg(self.stored_water - self.max)
        if runoff > 0 : 
            self.stored_water = self.max
        return runoff
    def remove_fraction(self,pet,w):
        exponent = ((pet - w) / self.max) * -1
        multiplier = (1.0 - e**exponent)
        amount_removed = self.stored_water * multiplier # This will return nan if pet or w are nan.
        if numpy.isnan(pet) == False and numpy.isnan(w) == False: # Keep the old soil water value if missing data. Number of missing PET and W values is tracked elsewhere.
            self.stored_water = self.stored_water - amount_removed
            if self.stored_water < 0 : self.stored_water = 0
        return amount_removed
              
def get_stn_info(station_names):
    import csv
    global station_info_path
    lat_dict = {}
    if station_info_path != '':
        inputname = station_info_path + 'all_stations.csv'
    else: inputname = 'all_stations.csv'
    
    input = open(inputname, encoding = "latin-1")
    reader = csv.reader(input)
    next(reader)
    for line in reader:
        station = line[3]
        latitude = line[4]
        id_no = line[2]
        elev = line[7]
        years_avail = line[9]
        if station in station_names:
            out_latitude = float(latitude)
            out_elevation = float(elev)
            out_stn_id = id_no 
            out_years_avail = years_avail
            
    input.close()
    out_years_avail = out_years_avail.split(';')
    return out_latitude, out_elevation, out_stn_id,out_years_avail

def swap_values(first,second):
    temp_val = first
    first = second
    second = temp_val
    return first, second

def make_int(val,low = True):
    try:
        return_val = int(val)
    except:
        if low == True: return_val = 0
        if low == False: return_val = 9999
    return return_val

def setup_start_and_end(firstyear,firstmonth,firstday,lastyear,lastmonth,lastday):
    first_year_val = make_int(firstyear); last_year_val = make_int(lastyear,low=False)
    first_month_val = make_int(firstmonth); last_month_val = make_int(lastmonth, low=False)
    first_day_val = make_int(firstday); last_day_val = make_int(lastday, low = False)

    if last_year_val < first_year_val: first_year_val,last_year_val = swap_values(first_year_val,last_year_val)
    if firstyear == lastyear and first_month_val > last_month_val : first_month_val, last_month_val = swap_values(first_month_val,last_month_val)
    if firstyear == lastyear and firstmonth == lastmonth and last_day_val < first_day_val : first_day_val, last_day_val = swap_values(first_day_val, last_day_val)
    too_much = False
    return first_year_val,first_month_val,first_day_val,last_year_val,last_month_val,last_day_val,too_much

def convert_dateobj_to_dates(dateobj):
    year = str(dateobj.year)
    month = str(dateobj.month)
    day = str(dateobj.day)
    if len(month) == 1: month = '0' + month
    if len(day) == 1: day = '0' + day
    return month,day,year

def convert_F_to_C(tval):
    if tval not in missing_val_list:
        tval = float(tval)
        retval = tval - 32
        retval = retval / 1.8
    else: retval = numpy.nan
    return retval

def convert_inches_to_mm(pval):
    pval = float(pval)
    if pval not in missing_val_list:
        retval = pval * 25.4
    else: retval = numpy.nan
    return retval
    
def lenfix(s):
    if len(s) == 1: s = '0' + s
    return s

def get_basic_values(valdict,this_index,station_type = 'GHCN'):
    global snow_density
    if this_index > len(valdict['tmax']) - 1: no_date = True
    else: no_date = False
    if not no_date:
        tmax = valdict['tmax'][this_index];tmax = convert_F_to_C(tmax)
        tmin = valdict['tmin'][this_index];tmin = convert_F_to_C(tmin)
        precip = valdict['precip'][this_index]; precip = convert_inches_to_mm(precip)
        if tmax > 75 : tmax = numpy.nan # Degrees C - Screen out unrealistic values
        if tmin > 75: tmin = numpy.nan
        if station_type == 'SNOTEL':
            snow = valdict['snow'][this_index]; snow = convert_inches_to_mm(snow)
            accum_precip = valdict['accum_precip'][this_index]; accum_precip = convert_inches_to_mm(accum_precip)
            return tmax,tmin,precip,snow,accum_precip
        elif station_type == 'GHCN': 
            snow = valdict['SNOW'][this_index];snow = convert_inches_to_mm(snow)
            snow = snow * snow_density # GHCN data has snow depth, not swe.
            return tmax,tmin,precip,snow
        elif station_type == 'RAWS': return tmax,tmin,precip
    else:
        if station_type == 'SNOTEL': return numpy.nan,numpy.nan,numpy.nan,numpy.nan,numpy.nan
        elif station_type == 'GHCN': return numpy.nan,numpy.nan,numpy.nan
        elif station_type == 'RAWS': return numpy.nan,numpy.nan,numpy.nan
        
    
def get_extra_RAWS_values(valdict,day):
    solar_rad = valdict['solar_rad'][int(day)-1]
    if solar_rad not in missing_val_list: solar_rad = float(solar_rad)
    rel_hum = valdict['rel_hum'][int(day)-1]
    if rel_hum not in missing_val_list : rel_hum = float(rel_hum)
    else: rel_hum = 'null' # This will trigger proxy estimation.
    wind_speed = float(valdict['wind_speed'][int(day)-1])
    if wind_speed not in missing_val_list: wind_speed = wind_speed *.44704 #Convert MPH to m/s
    else: wind_speed = float(1) # This is the default value used when wind data are not available.
    return solar_rad,rel_hum,wind_speed

def calculate_Ra(inverse_rel_distance,sunset_hour_angle,latitude,solar_declination): #CHECKS OK
    #Ra = extra-terrestrial radiation in MJ/m2/day. Equation 21 of Ch3 FAO doc.
    # Latitude (j) in radians. Positive for northern hemisphere and negative for southern hemisphere.
    # Sunset hour angle (Ws) in radians. Solar declination (d) in radians.
    Gsc = .0820 # Solar constant : MJ/m^2/day
    first_term = (float((24*60))/pi) * Gsc #37.586
    first_half_bracket_term = (numpy.sin(latitude) * numpy.sin(solar_declination)) * sunset_hour_angle
    second_half_bracket_term = (numpy.cos(latitude) * numpy.cos(solar_declination) * numpy.sin(sunset_hour_angle))
    full_bracket_term = first_half_bracket_term + second_half_bracket_term
    Ra = first_term * full_bracket_term * inverse_rel_distance
    return Ra

def calc_Rnl(tmax,tmin,Ea,Rs = 'null',Rso = 'null'): #CHECKS OK
    #Rnl = net long-wave radiation in MJ/m2/day. Equation 39 Ch3 FAO doc.
    #Ea = actual vapor pressure (Kpa). Rs = Solar radiation in MJ/m2/day as calc by equation 35.
    #Rso = clear sky radiation in MJ/m2/day as calculated by equation 37.
    if Rs == 'null' or Rso == 'null':
        Rs = 1
        Rso = 1
    relative_shortwave_radiation = Rs/Rso
    if relative_shortwave_radiation > 1 : relative_shortwave_radiation = 1
    tmax = tmax + 273.16;tmin = tmin + 273.16 # Convert to degrees Kelvin
    tmax = tmax**4;tmin=tmin**4
    left_bracket_term = (tmax+tmin)/2;left_bracket_term = left_bracket_term * .000000004903 # Stefan-Boltzmann constant
    middle_term = 0.34 - (.14 * numpy.sqrt(Ea))
    right_term = (1.35*relative_shortwave_radiation) - 0.35
    Rnl = left_bracket_term * middle_term * right_term
    return Rnl

def calc_Rn(Rnl,Rs): # CHECKS OK
    # Equation 40 of ch3 FAO doc.
    # Using estimate of Rns from equation 38 :> Rns = (1-a)*Rs. a is default set to 0.25 unless a regression has been done.
    Rns = 0.75* Rs
    Rn = Rns - Rnl
    return Rn

def calc_Rs(Ra,n,N): # Equation 35 CHECKS OK
    # Ra = extra-terrestrial radiation as calc by equation 21 in MJ/m2/day.
    # Rs = MJ/m2/day
    # n = actual duration of sunshine (hours)
    # N = maximum possible duration of sunshine (hours)
   
    a = 0.25 # Regression constant expressing fraction of extra-terrestrial radiation reaching the earth on overcast days
    b = 0.5 # a + b = fraction of extra-terrestrial radiation reaching earth on clear days.
    # These default values were taken from FAO Ch3 page 23. 
    ratio = float(n/N)
    bracket_term = (ratio * b) + a
    Rs = bracket_term * Ra
    return Rs

def calc_Rso(elevation,Ra): #CHECKS OK
    # Equation 37 in ch3 of FAO doc.
    # Ra as calc from equation 21.
    # Elevation in meters
    elevation = float(elevation)
    correction_term = (.00002 * elevation) + .75
    Rso = Ra * correction_term
    return Rso
    
def convert_decimal_latitude_to_radians(lat): #CHECKS OK
    lat = float(lat)
    retval = (pi/180) * lat
    return retval

def hargreaves_evapotrans(tmax,tmin,Ra):
    #Temperatures input as degrees C. Ra as MJ/m^2 /day
    tmax = float(tmax);tmin = float(tmin)
    if tmax not in missing_val_list and tmin not in missing_val_list:
        tmax = float(tmax);tmin = float(tmin)
        tmean = float((tmax + tmin) / 2)
        trange = tmax - tmin; sqrt_trange = numpy.sqrt(trange)
        offset_tmean = tmean + 17.8
        Et = .0023 * offset_tmean * sqrt_trange * Ra
        if Et < 0 : Et = 0
    else: Et = -99999
    return Et #mm/day
    
def hamon_evapotrans(tmax,tmin,daytime_length):
    es = calc_Es(tmax,tmin) # Lutz et al 2010
    tmean = float((tmax+tmin)/2)
    bottom = tmean+273.3
    bracket = es/bottom
    pet = 29.8*daytime_length * bracket
    return pet #mm / day
              
def calc_inverse_rel_distance(doy): #CHECKS OK
    doy = float(doy)
    bracket_term = ((2*pi)/365) * doy
    retval = (.033 * numpy.cos(bracket_term)) + 1
    return retval #radians

def calc_solar_declination(doy): #CHECKS OK
    doy = float(doy)
    bracket_term = (((2*pi)/365) * doy) - 1.39
    retval = .409 * numpy.sin(bracket_term)
    return retval # radians

def calc_sunset_hour_angle(latitude,solar_declination): # CHECKS OK
    #Both inputs need to be in radians
    bracket_term = numpy.tan(latitude) * -1 * numpy.tan(solar_declination)
    retval = numpy.arccos(bracket_term)
    return retval #radians
    
def calc_daylength(sunset_hour_angle): #equation 34 - CHECKS OK
    N = (24/pi)*sunset_hour_angle
    return N

def calc_Rs_from_t(tmax,tmin,Ra): #CHECKS OK
    #Equation 50
    t_range = tmax - tmin
    coeff = numpy.sqrt(t_range) * 0.16
    Rs = coeff * Ra
    return Rs

def calc_oudin_pet(tmax, tmin, Ra): # Equation 3 in Oudin et al. 2010
    # Since this is daily timestep, mdays = 1 and not included here.
    tmean = float((tmax+tmin)/2) 
    top = Ra * (tmean + 5) * .408
    bottom = 100
    PET = top/bottom
    PET = numpy.where(tmean > -5, PET, 0)
    return PET # mm / d
    
def calc_todays_PET(day_of_year,tmax,tmin,precip,station_latitude,elevation,Et_type):
    inverse_rel_distance = calc_inverse_rel_distance(day_of_year)
    solar_declination = calc_solar_declination(day_of_year)
    latitude = convert_decimal_latitude_to_radians(station_latitude)
    sunset_hour_angle = calc_sunset_hour_angle(latitude,solar_declination)
    Ra = calculate_Ra(inverse_rel_distance,sunset_hour_angle,latitude,solar_declination)
    daylength = calc_daylength(sunset_hour_angle)
    if Et_type == 'oudin':
        Et = calc_oudin_pet(tmax, tmin, Ra)
    elif Et_type == 'hargreaves': Et = hargreaves_evapotrans(tmax,tmin,Ra)
    elif Et_type == 'hamon': Et = hamon_evapotrans(tmax,tmin,daylength)
    elif Et_type == 'Penman_Montieth':
        G = 0 # Soil heat flux is assumed to be zero for daily calculations
        N = calc_daylength(sunset_hour_angle)
        if station_type != 'RAWS' or force_proxies == True:
            u2 = 2 # If no wind data available, wind speed is assumed to be 2 m/s.
            RHmean = 'null'
            Rso = calc_Rso(elevation,Ra) # Clear sky solar radiation
            Rs = calc_Rs_from_t(tmax,tmin,Ra) # Actual solar radiation derived from temperature
        else:
            n,RHmean,u2 = get_extra_RAWS_values(d[year][month],day)# If u2 is missing, it is set to 2 m/s. If RHmean is missing, it is set to null, which triggers non-RAWS proxy.
            if n in missing_val_list : n = N # If solar radiation data are missing, it is assumed that there were no clouds that day, i.e. all available daylight hours were sunny.
            #print 'rel hum = ' + str(RHmean)
            #print 'wind spd = ' + str(u2)
            Rso = calc_Rso(elevation,Ra)
            Rs = calc_Rs(Ra,n,N)   
        Es_minus_Ea,Ea = calc_Es_minus_Ea(tmax,tmin,RHmean) # If RHmean is null, alt methods will be called.
        Rnl = calc_Rnl(tmax,tmin,Ea,Rs,Rso) #The ratio of Rs/Rso constrained to 1.
        atmospheric_pressure = calc_atmospheric_pressure(elevation)
        gamma = calc_gamma(atmospheric_pressure)
        D = calc_D(tmax,tmin)
        Rn = calc_Rn(Rnl,Rs)
        Et = calc_Penman_Montieth_Eto(Rn,G,tmax,tmin,Es_minus_Ea,D,gamma,u2)
    return Et


def get_station_data(station,firstyear,lastyear):
    first_year_val,first_month_val,first_day_val,last_year_val,last_month_val,last_day_val,too_much = setup_start_and_end(firstyear,'1','1',lastyear,'12','31')
    if station_type == 'RAWS':
        d = read_daily_data.new_read_RAWS_hourly(station,first_year_val,last_year_val)
    if station_type == 'GHCN':
        d = read_daily_data.new_read_GHCN(station,first_year_val,last_year_val)
    if station_type == 'SNOTEL':
        d = read_daily_data.new_read_SNOTEL_tab(station,first_year_val,last_year_val)
    if station_type == 'Canadian': d = read_daily_data.new_read_canadian(station,first_year_val,last_year_val)
    yearlist = [str(x) for x in range(first_year_val,last_year_val + 1)]
    today = datetime.datetime(year = first_year_val,month = first_month_val,day = first_day_val)
    last_day = datetime.datetime(year = last_year_val, month = last_month_val, day = last_day_val)
    one_day = datetime.timedelta(days = 1)
    return d,yearlist,today,last_day,one_day
    
def nonneg(val):
    if val < 0 : return 0
    else: return val    
    
def onlyabsneg(val):
    if val > 0 : return 0
    else: return abs(val)

def calc_aet_and_runoff(soil_water,w,pet):
    if w > pet : 
        aet = pet
        excess = w - aet
        runoff = soil_water.add(excess) 
    else:
        runoff = 0
        delta_soil_water = soil_water.remove_fraction(pet,w)
        total_w_avail = delta_soil_water + w
        if pet > total_w_avail: aet = total_w_avail
        else: aet = pet
    return aet,runoff
    
def track_missing(md,month,year,values,ignore_snow = False): #values is a list of vals in paramlist order.
    paramlist = ['tmax','tmin','precip','accumsnow']
    if year not in md: md[year] = {}
    if month not in md[year]:
        md[year][month] = {}
        for param in paramlist: md[year][month][param] = 0
    i=0
    alert = False
    for param in paramlist:
        thisval = values[i]
        if numpy.isnan(thisval) == True: 
            md[year][month][param] +=1
            if md[year][month][param] > 3 : 
                if param == 'accumsnow' and ignore_snow == True: alert = False
                else: alert = True   
        i+=1
    return md, alert
        
def create_normals(d):
    yearlist = list(set(d['year']))
    exclude_keys = ['year','month','dates']
    out = {}
    i=0
    for year in yearlist:
        mask = d['year'] == year
        if numpy.sum(mask) < 365: continue
        if i == 0 : out['dates'] = d['dates'][mask][0:365]
        for key in d:
            if key in exclude_keys: continue
            if key not in out: out[key] = d[key][mask][0:365]
            else: out[key] = numpy.vstack((out[key],d[key][mask][0:365]))
        i+=1
    for key in out:
        if key in exclude_keys: continue
        try:
            out[key] = stats.nanmean(out[key],axis = 0)
        except:
            out[key] = numpy.nanmean(out[key],axis = 0)
            
    return out

def print_missing_table(md,print_enable = True, error_msg = True):
    #sys.stdout.flush()
    if print_enable: print("Content-Type: text/html\n")
    if error_msg: print('<p><b>Even with the most forgiving missing value setting, there is sometimes too little data to run a valid model. If for example, there are no snow measurements at all for 6 months in a year, the "very forgiving" choice will still yield no results. Use the missing value table below to choose a different set of years for your analysis. For example, if you chose 1967 - 2018, but you see that 1987 has a lot of missing values, try 1967 - 1986 and / or 1988 - 2018. Pick time periods that have <= 3 days with missing values in all parameters, if possible. It is ok to have missing values in the table for the remainder of this year. For example, if you are reading this in April 2018, it does not matter if the table shows missing values for May, June, etc in 2018.. . You can still choose 2018. It is also important to know that the year before your chosen time period must have complete data so that the model can equilibrate. For example, if you choose 2014 - 2018, then 2013 must also have complete data. If you are running the model on a RAWS, consider only the number of missing tmax, tmin and precip. The number of missing snow values for accumsnow on a RAWS does not matter because snow is estimated with a series of equations.</b></p>')
    set_delimiters(do_headers = False)
    yearlist,molist = make_yearlist(md)
    paramlist = ['tmax','tmin','precip','accumsnow']
    print(parastart)
    print(paraend)
    header =  rowstart + colstart + 'Month/Year' + colend
    for param in paramlist:
        header = header + colstart + 'Missing ' + param + colend
    header = header + rowend
    #print(parastart + '<p><b>Number of days missing in each month.</b></p>' + paraend)
    print(tablestart)
    print((header), end=' ')
    for year in yearlist:
        for month in molist:
            moyear = month + '/' + year
            outline = rowstart + colstart + moyear + colend
            
            for param in paramlist:
                try:
                    num_missing = str(md[year][month][param])
                except:
                    num_missing = 'all'
                outline = outline + colstart + num_missing + colend
            outline = outline + rowend
            print((outline), end=' ')
    

def WB(d,today,last_day,one_day,title,station,first_year_val,last_year_val,station_latitude,elevation,station_type = 'GHCN',Et_type = 'hamon',force_proxies = False,gddbase = 4.4444, soil_max_water = 100,forgive_snow = False,completely_ignore = False, sim_snow = False, aggbyyear = False, use_heatload = False,retrack_missing = False):
    heatload = simplified_heat_load(station_latitude)    
    outd=autodict(paramchecklist = ['Pack','tmax','tmin','p','t'])
    md = {} # Tracks missing values for each mo/year
    agddc = 0 #accumulated growing degree days in degrees C
    accum_precip = 0
    accumsnow = 0
    last_accumsnow = numpy.nan
    soil_water = soil_reservoir(soil_max_water) # mm
    i=0
    yearlist = []
    number_snow_replaced = 0
    this_current_year = str(datetime.datetime.now().year)
    while today != last_day + one_day:
        month,day,year = convert_dateobj_to_dates(today)
        if year not in yearlist : 
            yearlist.append(year)
            number_snow_replaced = 0
        month = lenfix(str(month))
        day = lenfix(str(day))
        date = month + '/' + day + '/' + year
        if i == 0:
            try:
                this_index = numpy.where(d['date'] == date)[0][0]
            except:
                today = today + one_day
                continue
        else:
            this_index +=1
        
        if month == '01' and day == '01' : agddc = 0
        day_of_year = today.timetuple().tm_yday
        if station_type == 'GHCN': 
            tmax,tmin,precip,accumsnowx = get_basic_values(d,this_index)
            if month == '10' and day == '01' :accum_precip = 0
            else: accum_precip = accum_precip + precip
        elif station_type == 'SNOTEL': tmax,tmin,precip,accumsnowx,accum_precip = get_basic_values(d,this_index,station_type = 'SNOTEL')
        elif station_type == 'RAWS': 
            tmax,tmin,precip = get_basic_values(d,this_index,station_type = 'RAWS')  
            if month == '10' and day == '01' :accum_precip = 0
            else: accum_precip = accum_precip + precip
        tmean = (tmax + tmin) /2     
        accumsnow = est_snow(accumsnow,tmean,precip)
        if numpy.isnan(accumsnow) == True and forgive_snow == True:
            accumsnow = last_accumsnow # Some observers don't record snow in the summer, particularly at places that have little snow to begin with. The user can decide to infill snow or not from the menu, setting the forgiving flag.
            md,alert = track_missing(md,month,year,[tmax,tmin,precip,numpy.nan],ignore_snow = True) # The last line will only infill snow if at least one value has been recorded at some point in the years considered because initial value of accumsnow was set numpy.nan.
            number_snow_replaced += 1
            if number_snow_replaced > 180 and year != this_current_year: 
                #print number_snow_replaced
                if retrack_missing == False: raise Exception('Too many snow values missing this year.')
                
        else: md,alert = track_missing(md,month,year,[tmax,tmin,precip,accumsnow])
        if alert == True and completely_ignore == False:
            if retrack_missing == False: raise Exception ('Too many missing values in ' + month + '/' + year)         
        
        gdd = nonneg(tmean - gddbase)
        if numpy.isnan(gdd) == False: agddc = agddc + gdd # Carry over previous days agdd if T data is missing.
        if i == 0 : 
            melt = 0
            daily_snow = 0
            rain = 0
            w = 0
        else:
            melt = onlyabsneg(accumsnow-last_accumsnow)
            daily_snow = nonneg(accumsnow - last_accumsnow)
            rain = nonneg(precip - daily_snow)
            w = rain + melt
        pet = calc_todays_PET(day_of_year,tmax,tmin,precip,station_latitude,elevation,Et_type)
        if use_heatload: pet = pet * heatload
        if accumsnow > 0 and numpy.isnan(pet) == False: pet = 0
        aet,runoff = calc_aet_and_runoff(soil_water,w,pet) # Also adds and removes water from the soil reservoir. 
        todays_soil_water = soil_water.stored_water
        deficit = nonneg(pet-aet)
        outd.add('month',month)
        outd.add('year',year)
        outd.add('w',w)
        outd.add('dailysnow',daily_snow)
        outd.add('rain',rain)
        outd.add('agddc',agddc)
        outd.add('melt',melt)
        outd.add('pet',pet)
        outd.add('dates',date)
        outd.add('Pack',accumsnow)
        outd.add('accumprecip',accum_precip)
        outd.add('p',precip)
        outd.add('t',tmean)
        outd.add('soilwater',todays_soil_water)
        outd.add('aet',aet)
        outd.add('runoff',runoff)
        outd.add('d',deficit)
        outd.add('tmax',tmax)
        outd.add('tmin',tmin)
        today = today + one_day
        last_accumsnow = accumsnow
        i+=1
    outd.make_arrays()
    outd.rectify_two_keys('d','soilwater')
    if station_type == 'SNOTEL': 
        outd.rectify_two_keys('p','Pack')
        outd.rectify_two_keys('p','melt')
        outd.rectify_two_keys('p','dailysnow')
    if retrack_missing == False: outd.raise_missing_errors() # If an important parameter is missing completely, this will force the program to fail even if the user has chosen to ignore missing values.
    if aggbyyear:
        aggd_data = create_normals(outd.d)
        return aggd_data
    else:
        return outd,md
    
def set_graph_spacing():
    matplotlib.rcdefaults()
    plt.style.use('classic')
    matplotlib.style.use('classic')
    matplotlib.rcParams['font.size']=12
    matplotlib.rcParams['legend.fontsize']= 10
    matplotlib.rcParams['figure.subplot.hspace']=0.15
    matplotlib.rcParams['figure.subplot.wspace']=0.30
    matplotlib.rcParams['xtick.labelsize']=13
    matplotlib.rcParams['ytick.labelsize']=13
    matplotlib.rcParams['axes.labelsize']=13
    matplotlib.rcParams['figure.subplot.left'] = 0.07
    matplotlib.rcParams['figure.subplot.right'] = 0.9
    matplotlib.rcParams['figure.subplot.top'] = 0.9
    matplotlib.rcParams['figure.subplot.bottom'] = 0.25
       
def set_delimiters(do_headers = True):
    global csv_output,station_type,year1,year2,colstart,colend,rowstart,rowend,tablestart,tableend,boldstart,boldend,parastart,paraend,table_type,sim_snow,use_heatload,forgiving,display_year1
    if csv_output == False:
        if do_headers:
            print("Content-Type: text/html\n")
            print('<p>')
            print('<a href = "https://climateanalyzer.biz/python/wb.py?station=' + station + '&year1=' + str(display_year1) + '&year2=' + str(year2)  + '&title=' + title + '&station_type=' + station_type + '&pet_type=' + pet_type + '&max_soil_water=' + str(max_soil_water) + '&snow_density=' + str(snow_density) + '&forgiving=' + str(forgiving) + '&table_type='+ table_type + '&sim_snow=' + str(sim_snow) + '&use_heatload=' + str(use_heatload) + '&csv_output=true&graph_table=table">Click Here To Download This Table As A Spreadsheet.</a>')
            print('</p>')
            print(' (There may be a slight delay before the download begins.)')
            print("<p>'-99999' = missing data. 'nan' = 'Not an number.' This indicates insufficent data for an accurate calculation.</p>")
        colstart = '<td>'
        colend = '</td>'
        rowstart = '<tr>'
        rowend = '</tr>'
        tablestart = '<TABLE border = "1" CELLPADDING = "5"><tr>'
        tableend = '</table>'
        boldstart = '<b>'
        boldend = '</b>'
        parastart = '<p>'
        paraend = '</p>'
    else:
        print("Content-Disposition: attachment; filename=" + station + "_wb.csv\n\n\r")
        colstart = ' '
        colend = ','
        rowstart = ' '
        rowend = '\r\n'
        tablestart = ' '
        tableend = ' '
        boldstart = ' '
        boldend = ' '
        parastart = ''
        paraend = '\r\n'
    
def calc_Penman_Montieth_Eto(Rn,G,tmax,tmin,Es_minus_Ea,D,gamma,u2=2): #CHECKS OK WITH EXAMPLE 17 CHAPTER 4.
    #Rn = net radiation (MJ/m2/day), G = soil heat flux density (MJ/m2/day),Tavg in degrees C
    #Es_minus_Ea = vapor pressure deficit in KPa, D = slope of vapor pressure curve
    #gamma = psychrometric constant (Kpa / degrees C)
    #u2 = wind speed at 2m height, set to 2 m/s by default
    tmax=float(tmax);tmin=float(tmin);Tavg = (tmax+tmin)/2
    topleft_term = .408*D*(Rn-G)
    corrected_tavg = Tavg+273
    topmiddle_term = (900/corrected_tavg)*gamma
    topright_term = u2*Es_minus_Ea
    overall_top_term = (topmiddle_term*topright_term) + topleft_term
    wind_correction = 0.34*u2
    bottomright_term = (1 + wind_correction) * gamma
    overall_bottom_term = D + bottomright_term
    Eto = overall_top_term / overall_bottom_term
    return Eto
       
def calc_gamma(atmospheric_pressure):#CHECKS OK
    # Gamma is the psychrometric constant used for Penman_Montieth Eto
    gamma = 0.665 * .001 * atmospheric_pressure
    return gamma

def calc_atmospheric_pressure(elevation): # CHECKS OK
    tr = .0065*elevation
    top = 293-tr
    inside = top/293
    right = inside**5.26
    atmos_pressure = 101.3*right
    return atmos_pressure

def calc_saturation_vapor_pressure(t): #CHECKS OK
    # This is Es for a given t
    # t = degrees C, equation 11 in FAO doc
    bracket_term = (17.27*t)/(t+237.3)
    right_term = numpy.exp(bracket_term)
    saturation_vapor_pressure = 0.6108 * right_term
    return saturation_vapor_pressure #kPa

def calc_Es(tmax,tmin): #CHECKS OK
    #Equation 12 in FAO doc
    # This is just saturation vapor pressure for Tmean
    tmax=float(tmax);tmin=float(tmin)
    e_tmax = calc_saturation_vapor_pressure(tmax)
    e_tmin = calc_saturation_vapor_pressure(tmin)
    Es = (e_tmax + e_tmin)/2
    return Es #kPa

def calc_D(tmax,tmin): #CHECKS OK
    #Slope of the vapor pressure curve
    # t is AVERAGE air temperature in degrees C
    # Equation 13 in FAO doc
    tmax = float(tmax);tmin=float(tmin)
    t = (tmax + tmin)/2
    bracket_term = (17.27*t)/(t+237.3)
    right_term = 0.6108 * numpy.exp(bracket_term)
    top_term = 4098*right_term
    bottom_term = (t+237.3)**2
    D = top_term/bottom_term
    return D

def calc_Ea_with_humidity_data(tmax,tmin,RHmean):#CHECKS OK
    #Equation 19 of chap3 in FAO doc
    #RHmean must be a %. It is reduced to decimal below.
    RHmean = float(RHmean)
    E_tmax = calc_saturation_vapor_pressure(tmax)
    E_tmin = calc_saturation_vapor_pressure(tmin)
    bracket_term = (E_tmax + E_tmin)/2
    left_term = RHmean / 100
    Ea = left_term * bracket_term
    return Ea
    
def calc_Es_minus_Ea(tmax,tmin,RHmean = 'null'): #CHECKS OK
    # Will use humidity data or estimate Ea by assuming that dewpoint is two degrees below daily Tmin.
    tmax = float(tmax);tmin=float(tmin)
    Es = calc_Es(tmax,tmin)
    if RHmean== 'null':
        tmin_adjusted = tmin - 2 # Suggested for arid regions by FAO. In the absence of humidity data, assume dewpoint is 2 degrees below Tmin.
        Ea = calc_saturation_vapor_pressure(tmin_adjusted)
    else:
        Ea = calc_Ea_with_humidity_data(tmax,tmin,RHmean)
    Es_minus_Ea = Es - Ea
    return Es_minus_Ea,Ea
 
def comparison_graph(Penman_data,Hamon_data,Hargreaves_data):
    cats = numpy.arange(len(Penman_data['Et']))
    labcats = cats[::10]
    labs = Penman_data['dates'][::10]
    plt.plot(cats,Penman_data['Et'],'black')
    plt.plot(cats,Hargreaves_data['Et'],'blue')
    plt.plot(cats,Hamon_data['Et'],'red')
    plt.title(title)
    plt.xticks(labcats,labs,rotation = 90)
    plt.legend(('Penman','Hargreaves','Hamon'))
    plt.ylabel('PET')
    plt.xlim(0,cats[-1])
    
def make_yearlist(md):
    yearlist =list(set(md.keys()))
    intyearlist = [int(x) for x in yearlist]
    intyearlist.sort()
    yearlist = [str(x) for x in intyearlist]
    molist = ['01','02','03','04','05','06','07','08','09','10','11','12']
    return yearlist,molist
    
def wb_daily_table(data_dict,md,title, printout = True):
    global station_type,forgive_snow,ignore_completely,display_year1
    #print display_year1
    #set_delimiters()
    yearlist,molist = make_yearlist(md)
    paramlist = ['tmax','tmin','precip','accumsnow']
    if printout == False: 
        fname = title.replace(' ','_') + '_water_balance.csv'
        outfile = open(fname,'w')
        report = outfile.write
    else:
        report = sys.stdout.write
    
    header = 'INDEX\tDATE\tP (MM)\tTMAX (C)\tTMIN (C)\tTMEAN (C)\tRAIN (MM)\tDAYS_SNOW (MM)\tACCUM_SNOWPACK (MM)\tMELT (MM)\tWATER_INPUT_TO_SOIL (MM)\tPET (MM)\tSOIL_WATER (MM)\tRUNOFF (MM)\tAET (MM)\tD (MM)\tAccum Growing Degree Days (C)'
    header = header.split('\t')
    header = [x for x in header if x != '']
    headline = rowstart + colstart 
    for item in header:
        headline = headline + item + colend + colstart
    headline = headline + rowend
    report(parastart + title + paraend)
    if completely_ignore: report(parastart + '**Missing value thresholds have been ignored. All calculations have been forced in months that exceed missing value criteria.**'+ paraend)
    elif not completely_ignore and not forgive_snow: report(parastart + 'Stringent missing value criteria in effect: all months have 3 or less missing values in all parameters.'+ paraend)
    if forgive_snow: report(parastart + '**Missing snow measurements have been replaced with the value from the previous day.' + paraend)    
    if station_type == 'GHCN': report(parastart + 'COOP data often contain many missing values. Please examine the missing value report that follows the main results in order to determine the validity of the model.' + paraend)
    
    report(tablestart)
    report(headline)
    
    i = 0
    for item in data_dict['dates']:
        this_year = item.split('/')[2]
        if int(this_year) < int(display_year1) :
            i+=1
            continue
        outline = rowstart + colstart + str(i) + colend + colstart + str(item) + colend + colstart + str(round(data_dict['p'][i],2)) + colend + colstart + str(round(data_dict['tmax'][i],2)) + colend + colstart + str(round(data_dict['tmin'][i],2)) + colend + colstart + str(round(data_dict['t'][i],2)) + colend + colstart + str(round(data_dict['rain'][i],2)) + colend + colstart + str(round(data_dict['dailysnow'][i],2)) + colend + colstart + str(round(data_dict['Pack'][i],2)) + colend 
        outline = outline + colstart + str(round(data_dict['melt'][i],2)) + colend + colstart + str(round(data_dict['w'][i],2)) + colend + colstart + str(round(data_dict['pet'][i],2)) + colend 
        outline = outline + colstart + str(round(data_dict['soilwater'][i],2)) + colend + colstart + str(round(data_dict['runoff'][i],2)) + colend + colstart + str(round(data_dict['aet'][i],2)) + colend + colstart + str(round(data_dict['d'][i],2)) + colend 
        outline = outline + colstart + str(round(data_dict['agddc'][i],2)) + colend + rowend
        i+=1
        report(outline),
    report(tableend)
    caveats()
    #if station_type == 'GHCN':    # Report missing values 
    report(parastart)
    report(paraend)
    header =  rowstart + colstart + 'Month/Year' + colend
    for param in paramlist:
        header = header + colstart + param + colend
    header = header + rowend
    report(parastart + '<p><b>Number of days missing in each month.</b></p>' + paraend)
    report(tablestart)
    report(header),
        
    for year in yearlist:
        for month in molist:
            moyear = month + '/' + year
            outline = rowstart + colstart + moyear + colend
            
            for param in paramlist:
                outline = outline + colstart + str(md[year][month][param]) + colend
            outline = outline + rowend
            report(outline),
    if printout == False: outfile.close()
    
    
def sparse_x_axis_labels(label_anchors,x_labels):
    bad_frac = True
    frac = 15
    while bad_frac:
        try:
            one_frac = int(round(len(label_anchors)/frac,0))
        except:
            break
        try:
            out_label_anchors = label_anchors[::one_frac]
            out_x_labels = x_labels[::one_frac]
            bad_frac = False
        except:
            frac = frac-1
    return out_label_anchors,out_x_labels   

def moving_average(x, N):
    y = numpy.zeros((len(x),))
    left_offset =  int(N/2)
    right_offset = int(N/2) 
    for ctr in range(len(x)):
         chunk = numpy.array(x[ctr-left_offset:ctr+right_offset+1])
         if len(chunk) < N :y[ctr] = numpy.nan
         elif numpy.sum(numpy.isnan(chunk)== True) > 3 : y[ctr] = numpy.nan
         else:y[ctr] = numpy.nansum(chunk)
         new_N = len(y[numpy.isfinite(y)])
    return y/N
    

def wb_graph(data_dict,title,aggregated_data = {},just_deficit = False, smooth_et = False):
    global img_format
    set_graph_spacing()
    overlap_color = (1.0,0.7411,0.7019)
    fig = plt.figure(figsize = (14,9))
    plt.title(title + ' Water Balance')
    ax1 = fig.add_subplot(1,1,1)
    if aggregated_data == {} : end_index = len(data_dict['dates']) + 1
    else: end_index = 731
    dates = data_dict['dates'][365:end_index] #  The first year is run through the model but not displayed.
    soilwater = data_dict['soilwater'][365:end_index]
    accumsnow = data_dict['Pack'][365:end_index]
    aet = data_dict['aet'][365:end_index]
    pet = data_dict['pet'][365:end_index]
    deficit = data_dict['d'][365:end_index]
    if smooth_et: 
        aet = moving_average(aet,14)
        pet = moving_average(pet,14)
    mv_deficit = moving_average(deficit,14)
    cats = numpy.arange(len(dates))
    if just_deficit == False:
        ax1.plot(cats,pet,'-',color = 'red',linewidth = 2,alpha = 1)
        ax1.plot(cats,aet,'-',color = 'blue',linewidth = 1,alpha = 1)
        ax1.fill_between(cats,pet,aet,where=pet>=aet,facecolor='orange',alpha=0.35,interpolate = False)
    else:
        ax1.plot(cats,mv_deficit,'-',color = 'orange',linewidth = 2,alpha = 1) 
        ax1.plot(cats,deficit,'-',color = 'gray',linewidth = 1,alpha = 0.25)
    if just_deficit == False: plt.ylabel('AET, PET and moisture deficit (mm)')
    else: plt.ylabel('Moisture deficit (mm)')
    if aggregated_data != {}:
        histpet = aggregated_data['pet']
        histaet = aggregated_data['aet']
        histdeficit = aggregated_data['d']
        histcats = numpy.arange(len(histaet))
        if just_deficit == False:
            ax1.plot(histcats,histpet,'--',color = 'red',linewidth = 3,alpha = 0.6)
            ax1.plot(histcats,histaet,'--',color = 'blue',linewidth = 2,alpha = 0.6)
            ax1.fill_between(histcats,histpet,histaet,where=histpet>=histaet,facecolor='magenta',alpha=0.15,interpolate = False)
        else:
            ax1.plot(histcats,histdeficit,'-',color = 'black',linewidth = 2,alpha = 1) 
        ax1.add_patch(patches.Rectangle((0.1, 0.1),0.5,0.5, color='orange',alpha=0.35 ))
        ax1.add_patch(patches.Rectangle((0.1, 0.1),0.5,0.5, color='magenta',alpha=0.15 ))
        ax1.add_patch(patches.Rectangle((0.1, 0.1),0.5,0.5, color=overlap_color,alpha=1.0 ))
        pet_string = "{o} - {t} (Historical) PET".format(o = cyear1,t=cyear2)
        deficit_string = "{o} - {t} Average daily deficit".format(o = cyear1,t=cyear2)
        if just_deficit == False and smooth_et == False: plt.legend(('Potential evapotranspiration (PET)','Actual evapotranspiration (AET)',pet_string,'Historical AET','Current moisture deficit','Historical moisture deficit'),loc = 8, bbox_to_anchor = (0.06,0.0,1,1),bbox_transform=plt.gcf().transFigure,handlelength = 3.0,ncol = 2)
        elif just_deficit == False and smooth_et == True:plt.legend(('14-Day moving average PET','14-Day moving average AET',pet_string,'Historical AET','14-day moisture deficit','Historical moisture deficit','Overlap historical/current deficit'),loc = 8, bbox_to_anchor = (0.06,0.0,1,1),bbox_transform=plt.gcf().transFigure,handlelength = 3.0,ncol = 2)
        else: plt.legend(('14-day moving average of current year moisture deficit', 'Current year daily deficit',deficit_string),loc = 8, bbox_to_anchor = (0.05,0.0,1,1),bbox_transform=plt.gcf().transFigure,handlelength = 3.0,ncol = 1)
        
    else:
        ax1.add_patch(patches.Rectangle((0.1, 0.1),0.5,0.5, color='orange',alpha=0.35 ))
        if just_deficit == False and smooth_et == False: plt.legend(('Potential evapotranspiration (PET)','Actual evapotranspiration (AET)','Moisture deficit'),loc = 8, ncol = 1,bbox_to_anchor = (0.06,0.0,1,1),bbox_transform=plt.gcf().transFigure,handlelength = 3.0,)
        elif just_deficit == False and smooth_et == True: plt.legend(('14-day moving avg. PET','14-day moving average AET','14-day moving average moisture deficit'),loc = 8, ncol = 1,bbox_to_anchor = (0.06,0.0,1,1),bbox_transform=plt.gcf().transFigure,handlelength = 3.0,)
        else: plt.legend(('14-day moving average of moisture deficit', 'Daily moisture deficit'),loc = 8, bbox_to_anchor = (0.05,0.0,1,1),bbox_transform=plt.gcf().transFigure,handlelength = 3.0,ncol = 1)
            
    ax2 = ax1.twinx()
    ax2.plot(cats,soilwater,'-',color = 'green',linewidth = 4,alpha = 0.7)
    plt.ylim(0,max_soil_water*1.05)
    plt.xlim(0,len(cats))
    plt.ylabel('Soil water (mm)')
    if aggregated_data != {}:
        ax2.plot(histcats,aggregated_data['soilwater'],'--',color = 'green',linewidth = 4,alpha = 0.7)
        plt.legend(('Est. soil water','Hist. est. soil water'),loc = 8, bbox_to_anchor = (0.34,0.01,1,1),bbox_transform=plt.gcf().transFigure,handlelength = 3.0,)
    else:
        plt.legend(('Est. soil water',),loc = 8, bbox_to_anchor = (0.34,0.01,1,1),bbox_transform=plt.gcf().transFigure,handlelength = 3.0,)
    label_cats,labels = sparse_x_axis_labels(cats,dates)
    plt.xticks(label_cats,labels,rotation = 90)
    plt.setp( ax1.xaxis.get_majorticklabels(), rotation=90 )
    if completely_ignore: plt.figtext(0.02,0.02,'**Missing value thresholds have been ignored.**',color = 'red')
    elif not completely_ignore and not forgive_snow :plt.figtext(0.02,0.02,'All months have 3 or less missing days in all parameters.',color = 'red') 
    if forgive_snow: plt.figtext(0.02,0.05,'**Missing snow measurements have been replaced.**' ,color = 'red')    
    if web == True:
        imgdata = BytesIO()
        if pub == False:fig.savefig(imgdata, format = 'png')
        else: fig.savefig(imgdata,dpi=300,format = img_format)
        return imgdata
    else:
        plt.show()
        
def est_snow(swe,todays_tavg,todays_precip):
    swe = melt_one_day(swe,todays_tavg)
    swe = accum_one_day(swe,todays_tavg,todays_precip)
    return swe
        
def melt_one_day(start_swe,tavg): # Hock 2003. 
    global melt_factor, melt_thresh_temperature   
    if tavg <= melt_thresh_temperature : return start_swe
    else:
        swe_delta = (tavg - melt_thresh_temperature) * melt_factor
        end_swe = start_swe - swe_delta
        if end_swe < 0 : end_swe = 0
        return end_swe
        
def accum_one_day(start_swe,tavg,precip): # Dingman et al. 2002., Lutz et al. 2010.
    global precip_fraction, melt_thresh_temperature
    if tavg <= melt_thresh_temperature : rain = 0.0
    elif tavg > (melt_thresh_temperature + 6) : rain = 1.0
    else:
        rain = precip_fraction * (tavg - melt_thresh_temperature)
    snow_increment = (1.0 - rain) * precip
    if snow_increment < 0 : snow_increment = 0
    end_swe = start_swe + snow_increment
    return end_swe

def offset_years(years_avail,year1,year2):
    year1_index = years_avail.index(year1)
    year2_index = years_avail.index(year2)
    if year1_index > year2_index : year1_index, year2_index = swap_values(year1_index,year2_index)
    if year1_index == 0: 
        year1_index = 1 # Need at least one year before display to run the model in order for carryover values to work. If there is < 3 yrs available the program will error out and prompt user to try again.
        if year2_index == 0 : year2_index = 1 # If too few years, error will occur below because index of 2 will exceed limit. 
        changed_year1 = True
    else: changed_year1 = False
   
    if year2_index > len(years_avail) -1 : # If the operation above tries to get years that don't exist...
        raise Exception ("Not enough years available.") # Raise an error and force program to stop. This will rarely occur.
    display_year1 = years_avail[year1_index] 
    year2 = years_avail[year2_index]
    year1 = years_avail[year1_index - 1] # Make the first calculation year one year earlier than first display year. If none of the boundary conditions above have been met, this will make first display year same as first year requested from menu.
    if year2_index - year1_index > 30 : too_many = True
    else: too_many = False
    return display_year1,year1,year2,too_many
    
def summ_by_month(month,year,data):
    out = {}
    keys =      ['dates', 'soilwater', 'tmin', 'aet', 'd',    'pet', 'tmax', 'dailysnow', 'melt',  'p',   'agddc', 'rain', 'w', 'year',   'runoff', 't',   'Pack']
    summ_func = ['null',   'mean',      'mean','sum', 'sum', 'sum', 'mean',  'sum',       'sum',   'sum', 'last',   'sum', 'sum', 'null', 'sum',    'mean', 'last'] 
    year_mask = data['year'] == year
    month_mask = data['month'] == month
    final_mask = year_mask * month_mask
    i=0
    for key in keys:
        this_data = data[key]
        this_data = this_data[final_mask]
        this_func = summ_func[i]
        if this_func == 'mean': 
            try:
                summ_val = stats.nanmean(this_data)
            except:
                summ_val = numpy.nanmean(this_data)
                
        elif this_func == 'sum': summ_val = numpy.nansum(this_data)
        elif this_func == 'last' : summ_val = this_data[-1]
        elif this_func == 'null' : 
            i+=1
            continue
        i+=1
        out[key] = summ_val
    return out
    
def caveats():
    print(parastart + 'In COOP and GHCN  stations, precipitation (both daily and accumulated) is an independent measurement from snow. This results in some locations showing more accumulated snow in swe than has been measured as total accumulated precip. ')
    print('Snow melt and accumulation are calculated from the snow measurements. "Rain" = total_precip (measured by the raingage) - dailysnowfall. Because of the independent measurements, some days will not show enough precipitation to account for the measured snow increase.')       
    print('For more details on how these small discrepancies are reconciled in the model, see our <a href="https://climateanalyzer.biz/ClimateAnalyzer_Water_Balance_Model.pdf">Methods</a>' + paraend)
    print('Units are mm and degrees C.')    
     
def wb_monthly_table(wb_data):
    global md
    keys =      ['soilwater', 'tmin', 'aet', 'd',    'pet', 'tmax', 'dailysnow', 'melt', 'p',   'agddc', 'rain', 'w',   'runoff', 't',   'Pack']
    monthlist = ['01','02','03','04','05','06','07','08','09','10','11','12']
    yearlist = list(set(wb_data.d['year']))
    intlist = [int(x) for x in yearlist]
    intlist.sort()
    yearlist = [str(x) for x in intlist]
    header = rowstart + colstart + 'Month' + colend + colstart + 'Soil Water (mm)' + colend + colstart + 'Tmin (C)' + colend + colstart + 'Actual Evapotranspiration (mm)' + colend + colstart + 'Deficit (mm)' + colend + colstart + 'Potential Evapotranspiration (mm)' + colend + colstart + 'Tmax (C)' + colend + colstart + 'Daily Snow (mm)' + colend + colstart + 'Snow Melt (mm)' + colend + colstart + 'Precipitation (mm)' + colend + colstart + 'Accumulated Growing Degree Days (C)' + colend + colstart + 'Rain (mm)' + colend + colstart + 'Water Input To Soil (mm)' + colend + colstart + 'Runoff (mm)' + colend + colstart + 'Tmean (C)' + colend + colstart + 'Snowpack (mm)' + colend
    header = header + rowend
    caveats()
    
    print(tablestart)
    print(header, end=' ')
    
    for year in yearlist[1:]:
        for month in monthlist:
            these_summaries = summ_by_month(month,year,wb_data.d)
            outline = rowstart + colstart + month + '/' + year + colend
            for key in keys:
                outline = outline + colstart + str(round(these_summaries[key],2)) + colend
            outline = outline + rowend
            print(outline, end=' ')
    print_missing_table(md, print_enable = False, error_msg = False)
            
def simplified_heat_load(station_latitude): 
    # Assumes that slope and aspect are zero, since there is no programmatic way to get these for weather station locations.
    latitude = convert_decimal_latitude_to_radians(station_latitude)
    hl = 0.339 + 0.808*numpy.cos(latitude)
    return hl
    
if __name__ == '__main__':
    melt_thresh_temperature = -6
    melt_factor = 0.35
    precip_fraction = 0.167
    sys.stderr = sys.stdout
    if web == True:
        form = cgi.FieldStorage()
        station = form['station'].value
        year1 = form['year1'].value
        try:
            year2 = form['year2'].value
        except:
            year2 = year1
        orig_year1 = year1
        orig_year2 = year2
        title = form['title'].value
        station_type = form['station_type'].value
        pet_type = form['pet_type'].value
        max_soil_water = float(form['max_soil_water'].value) #mm
        forgiving = form['forgiving'].value
        printout = True
        if forgiving.lower() == 'mild': 
            forgive_snow = True #Infill missing snow depth in GHCN data with the previous day's measurement.
            completely_ignore = False
        elif forgiving.lower() == 'very': 
            forgive_snow = True
            completely_ignore = True  #Ignore all missing data and calculate anyway
        else:
            forgive_snow = False
            completely_ignore = False
        try:
            csv_output = form['csv_output'].value
            if csv_output.lower() == 'true': csv_output = True
            else: csv_output = False
        except:
            csv_output = False
        try:
            snow_density = float(form['snow_density'].value) # For GHCN data conversion of snow depth to swe. Only applies to GHCN data.
        except:
            snow_density = 0.2
        
        try:
            graph_table = form['graph_table'].value
        except:
            graph_table = 'table'
        try:
            table_type = form['table_type'].value
        except:
            table_type = 'daily'
        try:
            force = form['force'].value
            if force.lower() == 'true': force = True
            else: force = False
        except:
            force = False
        try:
            sim_snow = form['sim_snow'].value
            if sim_snow.lower() == 'true': sim_snow = True
            else: sim_snow = False
        except:
            sim_snow = False
        try:
            cyear1 = form['cyear1'].value
        except:
            cyear1 = 'null'
        try:
            cyear2 = form['cyear2'].value
        except:
            cyear2 = 'null'
        try:
            just_deficit = form['just_deficit'].value
            if just_deficit.lower() == 'true' : just_deficit = True
            else: just_deficit = False
        except:
            just_deficit = False
        try:
            smooth_et = form['smooth_et'].value
            if smooth_et.lower() == 'true' : smooth_et = True
            else: smooth_et = False
        except:
            smooth_et = True
        try:
            use_heatload = form['use_heatload'].value
            if use_heatload.lower() == 'true' : use_heatload = True
            else: use_heatload = False
        except:
            use_heatload = True # This will be reversed if Penman Montieth WB is selected.
        try:
            img_format = form['format'].value
        except:
            img_format = 'png'
        try:
            pub = form['pub'].value
            if pub.lower() == 'true': pub = True
            else: pub = False
        except:
            pub = False
        

    else:
        station = 'westglacier'
        year1 = '2014'
        year2 = '2016'
        cyear1 = 'null'
        cyear2 = 'null'
        title = "West Glacier"
        station_type = 'GHCN'
        csv_output = True
        pet_type = "hamon"
        max_soil_water = 100.0 #mm
        printout = False # False will send data to a file. True sends it to sys.stdout
        snow_density = 0.2 # For GHCN data conversion of snow depth to swe
        graph_table = 'graph'
        forgiving = 'null' # For non-web testing purposes only. This flag is actually handled by the next two below.
        forgive_snow = True
        completely_ignore = True
        force = True # Ignore 30 year limit.
        sim_snow = False
        smooth_et = True
        just_deficit = True
        table_type = 'monthly'
        use_heatload = False
    if pet_type != 'hamon': use_heatload = False  # Penman Montieth already considers latitude and we don't consider slope and aspect here.
    station_type = station_type.upper()
    if cyear1 != 'null' and cyear2 != 'null' : year2 = year1 # The purpose of comparison graphs is compare only one year to normals.    
    if station_type == 'RAWS': sim_snow = True
    
    force_proxies = True # The code for supplemental RAWS data hasn't been updated to reflect the new memory structure yet. 4/28/2016.
    latitude,elevation, station_id,years_avail = get_stn_info([station])
    try:
        display_year1,year1,year2,too_many = offset_years(years_avail,year1,year2)
        good_years = True
    except:
        print("Content-Type: text/html\n")
        print('Choose a different set of years.')
        good_years = False
    too_many = False
    bad_wb = False
    sim_snow = True
    if good_years:    
        if too_many and force == False:
            print("Content-Type: text/html\n")
            print('You can only analyze 30 years or less at a time.')
        else:
            try:
                if station_type == 'GHCN': 
                    if '(' not in title: title = title + ' (' + station_id + ')'
                d,yearlist,today,last_day,one_day = get_station_data(station,year1,year2)
                try:
                    wb_data,md = WB(d,today,last_day,one_day,title,station,yearlist[0],yearlist[-1],latitude,elevation,station_type,pet_type,force_proxies,soil_max_water = max_soil_water,forgive_snow = forgive_snow,completely_ignore = completely_ignore, sim_snow = sim_snow,use_heatload = use_heatload) # PET is added as a monthly key to the data dictionary for the weather station
                    if cyear1 != 'null' and cyear2 != 'null': # If comparing to a historical period, call WB again to get the normals and set aggbyear = True.
                        d,yearlist,today,last_day,one_day = get_station_data(station,cyear1,cyear2)
                        aggregated_data = WB(d,today,last_day,one_day,title,station,yearlist[0],yearlist[-1],latitude,elevation,station_type,pet_type,force_proxies,soil_max_water = max_soil_water,forgive_snow = forgive_snow,completely_ignore = completely_ignore, sim_snow = sim_snow,use_heatload = use_heatload,aggbyyear=True) 
                    else: aggregated_data = {}
                except:
                    #print "Content-Type: text/html\n"
                    bad_wb = True
                    if cyear1 == 'null' and cyear2 == 'null':
                        d,yearlist,today,last_day,one_day = get_station_data(station,year1,year2)
                    else:
                        d,yearlist,today,last_day,one_day = get_station_data(station,cyear1,cyear2)
                    wb_data, md = wb_data,md = WB(d,today,last_day,one_day,title,station,yearlist[0],yearlist[-1],latitude,elevation,station_type,pet_type,force_proxies,soil_max_water = max_soil_water,forgive_snow = forgive_snow,completely_ignore = completely_ignore, sim_snow = sim_snow,use_heatload = use_heatload,retrack_missing = True)
                    print_missing_table(md)
                    
                if bad_wb == False:
                    if graph_table == 'table':
                        set_delimiters()
                        if table_type == 'daily': wb_daily_table(wb_data,md,title,printout = printout)
                        elif table_type == 'monthly': wb_monthly_table(wb_data)
                    else:
                        x = wb_graph(wb_data,title,aggregated_data = aggregated_data,just_deficit = just_deficit, smooth_et = smooth_et)
                        if pub == False:
                            x = x.getvalue()
                            #x = x.encode('base64').strip()
                            x = base64.encodebytes(x).decode()
                            html = html = """<html><body><title>ClimateAnalyzer Water Balance</title>"""
                            dl_link = '<a href="https://climateanalyzer.biz/python/wb.py?format=pdf&pub=true&station={station}&year1={year1}&year2={year2}&title={title}&station_type={station_type}&pet_type={pet_type}&max_soil_water={max_soil_water}&forgiving={forgiving}&graph_table=graph&force={force}&sim_snow={sim_snow}&cyear1={cyear1}&cyear2={cyear2}&just_deficit={just_deficit}">Download an editable vector pdf version of this graph.</a>'.format(station=station,year1=orig_year1,year2=orig_year2, title=title,station_type=station_type,pet_type=pet_type,max_soil_water=max_soil_water,forgiving=forgiving,force=force,sim_snow=sim_snow,cyear1=cyear1,cyear2=cyear2,just_deficit=str(just_deficit))
                            dl_link2 = '<a href="https://climateanalyzer.biz/python/wb.py?format=png&pub=true&station={station}&year1={year1}&year2={year2}&title={title}&station_type={station_type}&pet_type={pet_type}&max_soil_water={max_soil_water}&forgiving={forgiving}&graph_table=graph&force={force}&sim_snow={sim_snow}&cyear1={cyear1}&cyear2={cyear2}&just_deficit={just_deficit}">Download 300 DPI png version of this graph.</a>'.format(station=station,year1=orig_year1,year2=orig_year2, title=title,station_type=station_type,pet_type=pet_type,max_soil_water=max_soil_water,forgiving=forgiving,force=force,sim_snow=sim_snow,cyear1=cyear1,cyear2=cyear2,just_deficit=str(just_deficit))
                            tag = """<br><img src="data:image/png;base64,%s"/></body></html>""" %(x)
                            print("Content-Type: text/html\n")
                            html = html + dl_link + "&nbsp;" + "&nbsp;"+ "&nbsp;" + dl_link2  + tag
                            print(html)
                        else:
                            print('Content-Type: image/png')
                            if img_format == 'pdf': print('Content-Disposition: attachment; filename=wb.pdf\n')
                            else: print('Content-Disposition: attachment; filename=wb.png\n')
                            sys.stdout.flush()
                            x.seek(0)
                            outputstream = os.fdopen(sys.stdout.fileno(), 'wb')
                            outputstream.write(x.read())
                            outputstream.flush()
            except:
                print("Content-Type: text/html\n")
                print("<p>Unable to calculate water balance for this station. Try a different time period or relax the missing value forgiveness in the menus.</p>")
                print("<p>If one of the needed parameters is missing for an entire year, the model will refuse to run regardless of which missing value threshold is chosen.</p>")
               
        
           
    
      
