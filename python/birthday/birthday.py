#!/usr/bin/env python

print 'Enter your birthday information: '

month = input('Month (MM): ')
day = input('Day (DD): ')
year = input('Year (YYYY): ')

from datetime import date

birthday = date(year,month,day)

print
print 'You were born on: ' + birthday.strftime("%m-%d-%y")
print
print birthday.strftime('%d %b %Y was a %A on the %d day of %B')
print

now = date.today()
age = now - birthday

print 'You have lived   ' + str(age.days/365) + '  years &'
print '                 ' + str(age.days%365) + ' days '
print

birthday = date(now.year,month,day)

print 'This year, your birthday is on a: ' +birthday.strftime("%A")
print

togo = birthday - now

print 'You have ' + str(togo.days%365) + ' days until your next birthday'
print
