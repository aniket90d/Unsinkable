import datetime

now = datetime.datetime.now()
yr = now.year
mnth = now.month
day = now.day

day_list = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'twentyfirst', 'twentysecond', 'twentythird', 'twentyfourth', 'twentyfifth', 'twentysixth', 'twentyseventh', 'twentyeighth', 'twentyninth', 'thirtieth', 'thirtyfirst']
mnth_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

print "Hey there, World! \nHow are things today, on the", day_list[day - 1].title(), "day of", mnth_list[mnth - 1], yr, "?"
