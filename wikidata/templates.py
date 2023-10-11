KEYS = [
    "name",
    "wikidata_type",
    "description",
    "type",
    "prompt_fw",
    "prompt_bw",
    "few_shot_examples",
    "templates_fw",
    "templates_bw",
]

TEMPLATES = {
    "P17": {
        "name": "country",
        "wikidata_type": "WI",
        "description": "sovereign state that this item is in (not to be used for human beings)",
        "type": "many_to_one",
        "prompt_fw": "Please identify the country (present or historical) in which the following entities are located:",
        "prompt_bw": "Please identify an entity located in the following country (present or historical):",
        "few_shot_examples": [
            ("Stockholm", "Sweden"),
            ("London", "United Kingdom"),
            ("Harvard University", "United States of America"),
        ],
        "templates_fw": [
            "\"<key>\" is located in the following country: \"<value>\"",
            "\"<key>\" can be found in the following country: \"<value>\"",
            "\"<key>\" can be seen by visitors to the following country: \"<value>\"",
            "\"<key>\" is most closely associated with the following country: \"<value>\"",
        ],
        "templates_bw": [
            "\"<value>\" is the location of many entities, including \"<key>\"",
            "\"<value>\" is the sovereign state you'd visit to see, among other things, \"<key>\"",
        ],
    },
    "P22": {
        "name": "father",
        "wikidata_type": "WI",
        "description": "male parent of the subject. For stepfather, use \"stepparent\" (P3448)",
        "type": "many_to_one",
        "prompt_fw": "Please identify the father of the following individuals:",
        "prompt_bw": "Please identify an individual whose father was the following person:",
        "few_shot_examples": [
            ("George W. Bush", "George H. W. Bush"),
            ("Anna Wintour", "Charles Wintour"),
            ("Charles III", "Prince Philip"),
        ],
        "templates_fw": [
            "\"<key>\" was fathered by \"<value>\"",
            "The father of \"<key>\" is \"<value>\"",
            "The dad of \"<key>\" is \"<value>\"",
        ],
        "templates_bw": [
            "\"<value>\" is the father of \"<key>\"",
            "\"<value>\" is the dad of \"<key>\"",
        ],
    },
    "P25": {
        "name": "mother",
        "wikidata_type": "WI",
        "description": "female parent of the subject. For stepmother, use \"stepparent\" (P3448)",
        "type": "many_to_one",
        "prompt_fw": "Please identify the mother of the following individuals:",
        "prompt_bw": "Please identify an individual whose mother was the following person:",
        "few_shot_examples": [
            ("George W. Bush", "Barbara Bush"),
            ("Anna Wintour", "Eleanor Trego Baker"),
            ("Charles III", "Elizabeth II"),
        ],
        "templates_fw": [
            "<key> was birthed by <value>",
        ],
        "templates_bw": [
            "<value> is the mother of <key>",
            "<value> is the mom of <key>",
        ],
    },
    "P35": {
        "name": "head of state",
        "wikidata_type": "WI",
        "description": "official with the highest formal authority in a country/state",
        "type": "one_to_many",
        "prompt_fw": "Please identify a head of state of the following countries (present or historical):",
        "prompt_bw": "Please identify the country (present or historical) whose head of state was the following person:",
        "few_shot_examples": [
            ("United States of America", "Joseph Biden"),
            ("United Kingdom", "Rishi Sunak"),
            ("Germany", "Frank-Walter Steinmeier"),
        ],
        "templates_fw": [
            "<key> is governed by head of state <value>",
            "The head of state of <key> is <value>",
        ],
        "templates_bw": [
            "<value> is the head of state of <key>",
            "<value> presides over <key>",
            "<value> leads the government of <key>",
        ],
    },
    "P36": {
        "name": "capital",
        "wikidata_type": "WI",
        "description": "seat of government of a country, province, state or other type of administrative territorial entity",
        "type": "one_to_one",
        "prompt_fw": "Please identify the seat of government of the following entities:",
        "prompt_bw": "Please identify the place whose seat of government is the following city:",
        "few_shot_examples": [
            ("Westchester County, New York", "White Plains"),
            ("United Kingdom", "London"),
            ("Sweden", "Stockholm"),
        ],
        "templates_fw": [
            "The capital of <key> is the city <value>",
            "The seat of government of <key> is located in <value>",
        ],
        "templates_bw": [
            "<value> is the capital of <key>",
            "<value> is the seat of government of <key>",
        ],
    },
    "P37": {
        "name": "official language",
        "wikidata_type": "WI",
        "description": "official language",
        "type": "many_to_many",
        "prompt_fw": "Please identify an official language of the following entities:",
        "prompt_bw": "Please identify an entity whose official language is the following language:",
        "few_shot_examples": [
            ("Switzerland", "German"),
            ("United Kingdom", "English"),
            ("Sweden", "Swedish"),
        ],
        "templates_fw": [
            "One of the official languages of <key> is <value>",
            "<key> lists the following as an official language: <value>",
        ],
        "templates_bw": [
            "<value> is an official language of <key>",
            "<value> is one of the official languages of <key>",
        ]
    },
    "P38": {
        "name": "currency",
        "wikidata_type": "WI",
        "description": "currency used by item",
        "type": "many_to_one",
        "prompt_fw": "Please identify the currency of the following countries (present or historical):",
        "prompt_bw": "Please identify a country (present or historical) whose currency is the following currency:",
        "few_shot_examples": [
            ("Switzerland", "Swiss franc"),
            ("United Kingdom", "Pound sterling"),
            ("Sweden", "Swedish krona"),
        ],
        "templates_fw": [
            "<key> uses the currency <value>",
            "The currency of <key> is the <value>",
            "To buy something in <key>, you'll need to exchange your dollars for <value>",
        ],
        "templates_bw": [
            "The <value> is the currency in <key>",
            "The <value> can be spent in <key>",
        ],
    },
    "P50": {
        "name": "author",
        "wikidata_type": "WI",
        "description": "main creator(s) of a written work (use on works, not humans); use P2093 when Wikidata item is unknown or does not exist",
        "type": "many_to_one",
        "prompt_fw": "Please identify the author of the following written works:",
        "prompt_bw": "Please identify a written work whose author was the following person:",
        "few_shot_examples": [
            ("The Hobbit", "J.R.R Tolkien"),
            ("Cat's Cradle", "Kurt Vonnegut"),
            ("Paradise Lost", "John Milton"),
        ],
        "templates_fw": [
            "\"<key>\" was written by <value>",
            "\"<key>\" was authored by <value>",
            "\"<key>\" was created by <value>",
        ],
        "templates_bw": [
            "<value> is the author of \"<key>",
            "<value> wrote \"<key>",
        ],
    },
    "P57": {
        "name": "director",
        "wikidata_type": "WI",
        "description": "director(s) of film, TV-series, stageplay, video game or similar",
        "type": "many_to_one",
        "prompt_fw": "Please identify the director of the following works:",
        "prompt_bw": "Please identify a work directed by the following person:",
        "few_shot_examples": [
            ("Barry Lyndon", "Stanley Kubrick"),
            ("Taxi Driver", "Martin Scorsese"),
            ("The Fellowship of the Ring", "Peter Jackson"),
        ],
        "templates_fw": [
            "\"<key>\" was directed by <value>",
            "The director of \"<key>\" was <value>",
        ],
        "templates_bw": [
            "<value> was the director of \"<key>",
            "<value> directed \"<key>"
        ],
    },
    "P69": {
        "name": "educated at",
        "wikidata_type": "WI",
        "description": "educational institution attended by subject",
        "type": "many_to_many",
        "prompt_fw": "Please identify an educational institution attended by each of the following people:",
        "prompt_bw": "Please identify a person who attended the following educational institution:",
        "few_shot_examples": [
            ("Albert Einstein", "University of Zurich"),
            ("Barack Obama", "Harvard Law School"),
            ("Angela Merkel", "Leipzig University"),
        ],
        "templates_fw": [
            "<key> was educated at <value>",
            "<key> graduated from <value>",
        ],
        "templates_bw": [
            "<value> is an educational institution attended by <key>",
            "<value> was attended by <key>",
        ],
    },
    "P84": {
        "name": "architect",
        "wikidata_type": "WI",
        "description": "person or architectural firm responsible for designing this building",
        "type": "many_to_many",
        "prompt_fw": "Please identify an architect of the following buildings:",
        "prompt_bw": "Please identify a building designed by the following architect:",
        "few_shot_examples": [
            ("The Shard", "Renzo Piano"),
            ("The Guggenheim Museum Bilbao", "Frank Gehry"),
            ("The Sydney Opera House", "Jørn Utzon"),
        ],
        "templates_fw": [
            "<key> was designed by <value>",
            "The architect credited with designing <key> is <value>",
        ],
        "templates_bw": [
            "<value> is the primary architect of <key>",
        ],
    },
    "P102": {
        "name": "member of political party",
        "wikidata_type": "WI",
        "description": "the political party of which a person is or has been a member or otherwise affiliated",
        "type": "many_to_many",
        "prompt_fw": "Please identify a political party with which each of the following people has been affiliated:",
        "prompt_bw": "Please identify a person who has been affiliated with the following political party:",
        "few_shot_examples": [
            ("Barack Obama", "Democratic Party"),
            ("Angela Merkel", "Christian Democratic Union of Germany"),
            ("Vladimir Putin", "United Russia"),
        ],
        "templates_fw": [
            "<key> is a member of the political party <value>",
            "<key> is affiliated with the political party <value>",
        ],
        "templates_bw": [
            "<value> is the political party of <key>",
            "A notable member of <value> is <key>",
        ],
    },
    "P103": {
        "name": "native language",
        "wikidata_type": "WI",
        "description": "language or languages a person has learned from early childhood",
        "type": "many_to_many",
        "prompt_fw": "Please identify a native language spoken by each of the following people:",
        "prompt_bw": "Please identify a person who speaks the following language as their native language:",
        "few_shot_examples": [
            ("Barack Obama", "English"),
            ("Angela Merkel", "German"),
            ("Vladimir Putin", "Russian"),
        ],
        "templates_fw": [
            "The native language of <key> is <value>",
            "<key> is a native speaker of <value>",
        ],
        "templates_bw": [
            "<value> is the native language of <key>",
            "<value> is the first language of <key>",
        ],
    },
    "P113": {
        "name": "airline hub",
        "wikidata_type": "WI",
        "description": "airport that serves as a hub for an airline",
        "type": "many_to_many",
        "prompt_fw": "Please identify an airline hub for each of the following airlines:",
        "prompt_bw": "Please identify an airline whose hub is the following airport:",
        "few_shot_examples": [
            ("SAS", "Stockholm Arlanda Airport"),
            ("British Airways", "London Heathrow Airport"),
            ("Lufthansa", "Frankfurt Airport"),
        ],
        "templates_fw": [
            "<key> operates a hub at <value>",
        ],
        "templates_bw": [
            "<value> is a hub airport for <key>",
        ],
    },
    "P157": {
        "name": "killed by",
        "wikidata_type": "WI",
        "description": "person who killed the subject",
        "type": "many_to_one",
        "prompt_fw": "Please identify the killer of each of the following people:",
        "prompt_bw": "Please identify a person who was killed by the following person:",
        "few_shot_examples": [
            ("John F. Kennedy", "Lee Harvey Oswald"),
            ("Julius Caesar", "Marcus Junius Brutus"),
            ("Abraham Lincoln", "John Wilkes Booth"),
        ],
        "templates_fw": [
            "<key> was killed by <value>",
            "<key> was murdered by <value>",
        ],
        "templates_bw": [
            "<value> killed <key>",
            "<value> murdered <key>",
        ],
    },
    "P175": {
        "name": "performer",
        "wikidata_type": "WI",
        "description": "actor, musician, band or other performer associated with this role or musical work",
        "type": "many_to_one",
        "prompt_fw": "Please identify a performer of each of the following roles or musical works:",
        "prompt_bw": "Please identify one of the roles or musical works performed by the following performer:",
        "few_shot_examples": [
            ("James Bond", "Daniel Craig"),
            ("The Doctor", "Jodie Whittaker"),
            ("Here Comes the Sun", "The Beatles"),
        ],
        "templates_fw": [
            "\"<key>\" was performed by \"<value>\"",
            "\"<key>\" was played by \"<value>\"",
        ],
        "templates_bw": [
            "\"<value>\" is the performer of, among other things, \"<key>\"",
            "\"<value>\" played \"<key>\"",
        ],
    },
    "P177": {
        "name": "crosses",
        "wikidata_type": "WI",
        "description": "obstacle (body of water, road, railway...) which this bridge (ferry, ford) crosses over or this tunnel goes under",
        "type": "many_to_one",
        "prompt_fw": "Please identify the obstacle each of the following bridges:",
        "prompt_bw": "Please identify a bridge that crosses the following obstacle:",
        "few_shot_examples": [
            ("Golden Gate Bridge", "Golden Gate"),
            ("Brooklyn Bridge", "East River"),
            ("Sydney Harbour Bridge", "Sydney Harbour"),
        ],
        "templates_fw": [
            "<key> crosses <value>",
            "<key> spans <value>",
        ], 
        "templates_bw": [
            "<value> is crossed by <key>",
            "<value> is spanned by <key>",
            "Travelers can cross <value> with <key>",
        ],
    },
    "P184": {
        "name": "doctoral advisor",
        "wikidata_type": "WI",
        "description": "person who supervised the doctorate or PhD thesis of the subject",
        "type": "many_to_many",
        "prompt_fw": "Please identify a doctoral advisor of each of the following people:",
        "prompt_bw": "Please identify a person whose doctoral advisor was the following person:",
        "few_shot_examples": [
            ("Albert Einstein", "Alfred Kleiner"),
            ("Richard Feynman", "John Archibald Wheeler"),
            ("Angela Merkel", "Lutz Zülicke"),
        ],
        "templates_fw": [
            "<key> was advised by <value>",
            "At grad school, <key> was supervised by <value>",
        ],
        "templates_bw": [
            "<value> was the doctoral advisor of <key>",
            "<value> supervised the PhD thesis of <key>",
        ],
    },
    "P206": {
        "name": "located in or next to body of water",
        "wikidata_type": "WI",
        "description": "located in or next to body of water",
        "type": "many_to_one",
        "prompt_fw": "Please identify the body of water next to which each of the following is located:",
        "prompt_bw": "Please identify a place located next to the following body of water:",
        "few_shot_examples": [
            ("London", "River Thames"),
            ("Sydney Opera House", "Sydney Harbour"),
            ("The Statue of Liberty", "Hudson River"),
        ],
        "templates_fw": [
            "<key> is located closest to the body of water <value>",
        ],
        "templates_bw": [
            "<value> is the body of water closest to <key>",
        ],
    },
    "P241": {
        "name": "military branch",
        "wikidata_type": "WI",
        "description": "branch to which this military unit, award, office, or person belongs, e.g. Royal Navy",
        "type": "many_to_one",
        "prompt_fw": "Please identify the military branch (present or historical) with which each of the following is associated:",
        "prompt_bw": "Please identify a military branch (present or historical) associated with the following:",
        "few_shot_examples": [
            ("Dwight D. Eisenhower", "United States Army"),
            ("George Washington", "Continental Army"),
            ("Napoleon Bonaparte", "French Army"),
        ],
        "templates_fw": [
            "The military branch of <key> is <value>",
            "The branch of the military most closely associated with <key> is <value>",
        ],
        "templates_bw": [
            "<value> is the military branch of <key>",
            "<value> is the military branch most closely associated with <key>",
        ],
    },
    "P344": {
        "name": "director of photography",
        "wikidata_type": "WI",
        "description": "person responsible for the framing, lighting, and filtration of the subject work",
        "type": "many_to_many",
        "prompt_fw": "Please identify a director of photography of the following works:",
        "prompt_bw": "Please identify a work whose director of photography was the following person:",
        "few_shot_examples": [
            ("Barry Lyndon", "John Alcott"),
            ("Taxi Driver", "Michael Chapman"),
            ("The Fellowship of the Ring", "Andrew Lesnie"),
        ],
        "templates_fw": [
            "<key> was filmed by <value>",
            "<key> was shot by <value>",
            "The director of photography for <key> was <value>",
        ],
        "templates_bw": [
            "<value> is the director of photography for <key>",
            "<value> filmed <key>",
            "<value> shot <key>",
        ],
    },
    "P364": {
        "name": "original language of work",
        "wikidata_type": "WI",
        "description": "language in which a film or a performance work was originally created. Deprecated for written works and songs; use P407 (\"language of work or name\") instead.",
        "type": "many_to_one",
        "prompt_fw": "Please identify the original language of each of the following works:",
        "prompt_bw": "Please identify a work originally created in the following language:",
        "few_shot_examples": [
            ("The Hobbit", "English"),
            ("Kalevala", "Finnish"),
            ("One Hundred Years of Solitude", "Spanish"),
        ],
        "templates_fw": [
            "<key> was originally created in the following language: <value>",
            "<key> was originally written in the following language: <value>",
        ],
        "templates_bw": [
            "<value> is the original language of <key>",
        ],
    },
    "P376": {
        "name": "located on astronomical body",
        "wikidata_type": "WI",
        "description": "single astronomical body on which features or places are situated",
        "type": "many_to_one",
        "prompt_fw": "Please identify the astronomical body on which the following features or places are situated:",
        "prompt_bw": "Please identify a feature or place situated on the following astronomical body:",
        "few_shot_examples": [
            ("Mount Everest", "Earth"),
            ("Mare Nectaris", "Moon"),
            ("The Great Red Spot", "Jupiter"),
        ],
        "templates_fw": [
            "<key> is located on the astronomical body <value>",
            "<key> can be seen on the surface of <value>",
        ],
        "templates_bw": [
            "<value> is the astronomical body on which the following feature or place is located: <key>",
        ],
    },
}

for k in TEMPLATES:
    assert(all([key in KEYS for key in TEMPLATES[k]]))
    assert(all([key in TEMPLATES[k] for key in KEYS]))