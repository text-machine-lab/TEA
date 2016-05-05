# regex patterns for temporal expressions
FS = "\t";

unit = "^(second(s?)|minute(s?)|hour(s?)|day(s?)|week(s?)|month(s?)|semester(s?)|year(s?)|decade(s?)|decennial(s?)|century|centuries|millennium(s?)|millenia|trimester(s?))\\s?$";

parts_of_the_day = "^(morning(s?)|afternoon(s?)|noon(s?)|midday(s?)|evening(s?)|night(s?)|midnight(s?)|overnight(s?))\\s?$";

day = "^(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon\\.?|tue\\.?|wed\\.?|thu\\.?|fri\\.?|sat\\.?|sun\\.?)\\s?$";

month = "^(january|february|march|april|may|june|july|august|september|october|november|december|jan\\.?|feb\\.?|mar\\.?|apr\\.?|may\\.?|jun\\.?|jul\\.?|aug\\.?|sep\\.?|sept\\.?|oct\\.?|nov\\.?|dec\\.?)\\s?$";

season = "^(spring(s?)|summer(s?)|autumn(s?)|fall(s?)|winter(s?))\\s?$";

number = "^(0?[1-9]|[1-2][0-9]|30|31)\\s?$";

yy = "^('[1-9][0-9]|[1-2][0-9][0-9][0-9])\\s?$";

time = "^(([1-9]|[0-1][0-9]|20|21|22|23|24)([:.,]([0-5][0-9]|60)([:.,][0-5][0-9]60)?))\\s?$";

duration = "^([0-9][0-9]?|[0-9][0-9]*h[0-9][0-9]*)'\\s?$";

cardinal_number = "^(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|hundreds|thousand|thousands|million|millions|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|twenty-six|twenty-seven|twenty-eight|twenty-nine|thirty-one|thirty-two|thirty-three|thirty-four|thirty-five|thirty-six|thirty-seven|thirty-eight|thirty-nine|forty-one|forty-two|forty-three|forty-four|forty-five|forty-six|forty-seven|forty-eight|forty-nine|fifty-one|fifty-two|fifty-three|fifty-four|fifty-five|fifty-six|fifty-seven|fifty-eight|fifty-nine|sixty-one|sixty-two|sixty-three|sixty-four|sixty-five|sixty-six|sixty-seven|sixty-eight|sixty-nine|seventy-one|seventy-two|seventy-three|seventy-four|seventy-five|seventy-six|seventy-seven|seventy-eight|seventy-nine|eighty-one|eighty-two|eighty-three|eighty-four|eighty-five|eighty-six|eighty-seven|eighty-eight|eighty-nine|ninety-one|ninety-two|ninety-three|ninety-four|ninety-five|ninety-six|ninety-seven|ninety-eight|ninety-nine)\\s?$";

ordinal_number = "^(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|twenty-first|twenty-second|twenty-third|twenty-fourth|twenty-fifth|twenty-sixth|twenty-seventh|twenty-eighth|twenty-ninth|thirty-first)\\s?$";

adverbs = "^(today|yesterday|tomorrow|tonight|tonite|now|then|previously|formerly|recently|currently|contemporarily|prehistorically|lately|hourly|nightly|fortnightly|daily|weekly|monthly|yearly|annually|seasonally|quarterly|later|ago|once|soon|before|earlier|after|afterwards|a.m.|p.m.)\\s?$";

signal_words = "^(on|in|at|from|to|before|after|during|before|after|while|when|until|for|since|as|initially|whenever|subsequently|'s|follows|if|by|through|over|already|ended|previously|within|later|earlier|then|once|still|following|meanwhile|into|followed|former|formerly|meantime|simultaneously|thereafter|next|concurrently|twice)\\s?$";

names = "^(daybreak|sunrise|daylight|sun-up|dusk|once-a-year|year-end|year-long|twelve-month|beginning|start|biannual|biennial|semiannual|twice-yearly|biannually|contemporary|quarter|quarters|date|after|dozen|dozens|epoch|epochs|era|eras|age|ages|period|periods|time|times|span|spans|stage|stages|former|end|more|future|local|beginnings|lustrum|moment|moments|nineties|pair|couple|past|period|periods|lunch|previous|post|after|early|prehistoric|present|first|next|recent|delay|past|last|current|this|twilight|nightfall|sunset|sundown|eve|eventide|gloaming|evenfall|nighttime|darkness|dark|generation|springtime|week-end|weekend|weekends|week-ends|season|seasons|seasonal|time|times|christmas|easter)\\s?$";

set_pattern = "^(each|every)\\s?$";

