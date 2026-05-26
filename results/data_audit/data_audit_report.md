# Data Audit Report

- Generated: `2026-05-26T09:17:42.923106+00:00`
- Primary dataset: `data/processed/grammar_correction_full.jsonl`
- Rows: `2018`
- Cleanliness score: `99.22/100`
- Training readiness: `Moderate`

## Dataset Inventory

| path | classification | size | examples |
|---|---|---|---|
| data/processed/grammar_correction_full.jsonl | primary-dataset | 584.6 KB | 2018 |
| data/processed/grammar_correction_metadata.json | dataset-metadata | 5.9 KB | n/a |
| data/processed/grammar_correction_test.jsonl | primary-dataset | 61.5 KB | 202 |
| data/processed/grammar_correction_train.jsonl | primary-dataset | 496.6 KB | 1612 |
| data/processed/grammar_correction_validation.jsonl | primary-dataset | 62.8 KB | 204 |
| data/prompt_registry.json | support-metadata | 1.4 KB | 3 |
| data/raw/grammar_correction.csv | primary-dataset | 248.8 KB | 2018 |
| data/raw/manifest.json | support-metadata | 414 B | n/a |
| data/sample/grammar_rules.txt | support-text | 3.2 KB | 30 |
| data/sample/sample_sentences.txt | support-text | 1.9 KB | 20 |
| data/vector_store/chunks.json | support-artifact | 5.4 KB | 30 |
| data/vector_store/metadata.json | support-artifact | 186 B | n/a |

## Dataset Schemas

### `data/processed/grammar_correction_full.jsonl`

- Classification: `primary-dataset`
- File size: `584.6 KB`
- Examples: `2018`
- Input column: `original`
- Corrected column: `corrected`
- Label columns: `error_type`
- Metadata columns: `serial_number, source`

| column | dtype |
|---|---|
| serial_number | int64 |
| error_type | object |
| original | object |
| corrected | object |
| references | object |
| source | object |

| serial_number | error_type | original | corrected | references | source |
|---|---|---|---|---|---|
| 1 | Verb Tense Errors | I goes to the store everyday. | I go to the store everyday. | ['I go to the store everyday.'] | grammar_correction.csv |
| 2 | Verb Tense Errors | They was playing soccer last night. | They were playing soccer last night. | ['They were playing soccer last night.'] | grammar_correction.csv |
| 3 | Verb Tense Errors | She have completed her homework. | She has completed her homework. | ['She has completed her homework.'] | grammar_correction.csv |
| 4 | Verb Tense Errors | He don't know the answer. | He doesn't know the answer. | ["He doesn't know the answer."] | grammar_correction.csv |
| 5 | Verb Tense Errors | The sun rise in the east. | The sun rises in the east. | ['The sun rises in the east.'] | grammar_correction.csv |
| 6 | Verb Tense Errors | I am eat pizza for lunch. | I am eating pizza for lunch. | ['I am eating pizza for lunch.'] | grammar_correction.csv |
| 7 | Verb Tense Errors | The students studies for the exam. | The students study for the exam. | ['The students study for the exam.'] | grammar_correction.csv |
| 8 | Verb Tense Errors | The car need to be repaired. | The car needs to be repaired. | ['The car needs to be repaired.'] | grammar_correction.csv |
| 9 | Verb Tense Errors | She will goes to the party tonight. | She will go to the party tonight. | ['She will go to the party tonight.'] | grammar_correction.csv |
| 10 | Verb Tense Errors | They watches the movie together. | They watch the movie together. | ['They watch the movie together.'] | grammar_correction.csv |

### `data/processed/grammar_correction_metadata.json`

- Classification: `dataset-metadata`
- File size: `5.9 KB`
- Examples: `n/a`
- Input column: `None`
- Corrected column: `None`
- Label columns: `error_type_distribution, split_counts_by_error_type`
- Metadata columns: `none`

| column | dtype |
|---|---|
| dataset_name | str |
| source_file | str |
| random_seed | int |
| total_rows | int |
| fields | list |
| split_counts | dict |
| error_type_distribution | dict |
| split_counts_by_error_type | dict |
| output_files | dict |

| dataset_name | source_file | random_seed | total_rows | fields | split_counts | error_type_distribution | split_counts_by_error_type | output_files |
|---|---|---|---|---|---|---|---|---|
| grammar_correction | c:\Users\parva\OneDrive\Desktop\Project\Grammer Autocorrector\data\raw\grammar_correction.csv | 42 | 2018 | ['serial_number', 'error_type', 'original', 'corrected', 'references', 'source', 'split'] | {'train': 1612, 'validation': 204, 'test': 202} | {'Abbreviation Errors': 50, 'Agreement in Comparative and Superlative Forms': 49, 'Ambiguity': 50, 'Article Usage': 100, 'Capitalization Errors': 40, 'Clichés': 48, 'Conjunction Misuse': 49, 'Contractions Errors': 49, 'Ellipsis Errors': 49, 'Faulty Comparisons': 49, 'Gerund and Participle Errors': 50, 'Inappropriate Register': 49, 'Incorrect Auxiliaries': 50, 'Infinitive Errors': 49, 'Lack of Parallelism in Lists or Series': 50, 'Mixed Conditionals': 49, 'Mixed Metaphors/Idioms': 50, 'Modifiers Misplacement': 46, 'Negation Errors': 50, 'Parallelism Errors': 49, 'Passive Voice Overuse': 49, 'Preposition Usage': 95, 'Pronoun Errors': 47, 'Punctuation Errors': 60, 'Quantifier Errors': 48, 'Redundancy/Repetition': 20, 'Relative Clause Errors': 51, 'Run-on Sentences': 40, 'Sentence Fragments': 40, 'Sentence Structure Errors': 103, 'Slang, Jargon, and Colloquialisms': 50, 'Spelling Mistakes': 100, 'Subject-Verb Agreement': 100, 'Tautology': 50, 'Verb Tense Errors': 100, 'Word Choice/Usage': 40} | {'train': {'Abbreviation Errors': 40, 'Agreement in Comparative and Superlative Forms': 39, 'Ambiguity': 40, 'Article Usage': 80, 'Capitalization Errors': 32, 'Clichés': 38, 'Conjunction Misuse': 39, 'Contractions Errors': 39, 'Ellipsis Errors': 39, 'Faulty Comparisons': 39, 'Gerund and Participle Errors': 40, 'Inappropriate Register': 39, 'Incorrect Auxiliaries': 40, 'Infinitive Errors': 39, 'Lack of Parallelism in Lists or Series': 40, 'Mixed Conditionals': 39, 'Mixed Metaphors/Idioms': 40, 'Modifiers Misplacement': 37, 'Negation Errors': 40, 'Parallelism Errors': 39, 'Passive Voice Overuse': 39, 'Preposition Usage': 76, 'Pronoun Errors': 38, 'Punctuation Errors': 48, 'Quantifier Errors': 38, 'Redundancy/Repetition': 16, 'Relative Clause Errors': 41, 'Run-on Sentences': 32, 'Sentence Fragments': 32, 'Sentence Structure Errors': 82, 'Slang, Jargon, and Colloquialisms': 40, 'Spelling Mistakes': 80, 'Subject-Verb Agreement': 80, 'Tautology': 40, 'Verb Tense Errors': 80, 'Word Choice/Usage': 32}, 'validation': {'Abbreviation Errors': 5, 'Agreement in Comparative and Superlative Forms': 5, 'Ambiguity': 5, 'Article Usage': 10, 'Capitalization Errors': 4, 'Clichés': 5, 'Conjunction Misuse': 5, 'Contractions Errors': 5, 'Ellipsis Errors': 5, 'Faulty Comparisons': 5, 'Gerund and Participle Errors': 5, 'Inappropriate Register': 5, 'Incorrect Auxiliaries': 5, 'Infinitive Errors': 5, 'Lack of Parallelism in Lists or Series': 5, 'Mixed Conditionals': 5, 'Mixed Metaphors/Idioms': 5, 'Modifiers Misplacement': 5, 'Negation Errors': 5, 'Parallelism Errors': 5, 'Passive Voice Overuse': 5, 'Preposition Usage': 10, 'Pronoun Errors': 5, 'Punctuation Errors': 6, 'Quantifier Errors': 5, 'Redundancy/Repetition': 2, 'Relative Clause Errors': 5, 'Run-on Sentences': 4, 'Sentence Fragments': 4, 'Sentence Structure Errors': 10, 'Slang, Jargon, and Colloquialisms': 5, 'Spelling Mistakes': 10, 'Subject-Verb Agreement': 10, 'Tautology': 5, 'Verb Tense Errors': 10, 'Word Choice/Usage': 4}, 'test': {'Abbreviation Errors': 5, 'Agreement in Comparative and Superlative Forms': 5, 'Ambiguity': 5, 'Article Usage': 10, 'Capitalization Errors': 4, 'Clichés': 5, 'Conjunction Misuse': 5, 'Contractions Errors': 5, 'Ellipsis Errors': 5, 'Faulty Comparisons': 5, 'Gerund and Participle Errors': 5, 'Inappropriate Register': 5, 'Incorrect Auxiliaries': 5, 'Infinitive Errors': 5, 'Lack of Parallelism in Lists or Series': 5, 'Mixed Conditionals': 5, 'Mixed Metaphors/Idioms': 5, 'Modifiers Misplacement': 4, 'Negation Errors': 5, 'Parallelism Errors': 5, 'Passive Voice Overuse': 5, 'Preposition Usage': 9, 'Pronoun Errors': 4, 'Punctuation Errors': 6, 'Quantifier Errors': 5, 'Redundancy/Repetition': 2, 'Relative Clause Errors': 5, 'Run-on Sentences': 4, 'Sentence Fragments': 4, 'Sentence Structure Errors': 11, 'Slang, Jargon, and Colloquialisms': 5, 'Spelling Mistakes': 10, 'Subject-Verb Agreement': 10, 'Tautology': 5, 'Verb Tense Errors': 10, 'Word Choice/Usage': 4}} | {'full': 'c:\\Users\\parva\\OneDrive\\Desktop\\Project\\Grammer Autocorrector\\data\\processed\\grammar_correction_full.jsonl', 'train': 'c:\\Users\\parva\\OneDrive\\Desktop\\Project\\Grammer Autocorrector\\data\\processed\\grammar_correction_train.jsonl', 'validation': 'c:\\Users\\parva\\OneDrive\\Desktop\\Project\\Grammer Autocorrector\\data\\processed\\grammar_correction_validation.jsonl', 'test': 'c:\\Users\\parva\\OneDrive\\Desktop\\Project\\Grammer Autocorrector\\data\\processed\\grammar_correction_test.jsonl'} |

### `data/processed/grammar_correction_test.jsonl`

- Classification: `primary-dataset`
- File size: `61.5 KB`
- Examples: `202`
- Input column: `original`
- Corrected column: `corrected`
- Label columns: `error_type`
- Metadata columns: `serial_number, source, split`

| column | dtype |
|---|---|
| serial_number | int64 |
| error_type | object |
| original | object |
| corrected | object |
| references | object |
| source | object |
| split | object |

| serial_number | error_type | original | corrected | references | source | split |
|---|---|---|---|---|---|---|
| 1721 | Abbreviation Errors | My bro just joined the USMC yday. | My brother just joined the United States Marine Corps yesterday. | ['My brother just joined the United States Marine Corps yesterday.'] | grammar_correction.csv | test |
| 1727 | Abbreviation Errors | The doc prescribed me NSAIDs for my pain. | The doctor prescribed me nonsteroidal anti-inflammatory drugs for my pain. | ['The doctor prescribed me nonsteroidal anti-inflammatory drugs for my pain.'] | grammar_correction.csv | test |
| 1735 | Abbreviation Errors | The atm in the mall is out of svc. | The automated teller machine in the mall is out of service. | ['The automated teller machine in the mall is out of service.'] | grammar_correction.csv | test |
| 1737 | Abbreviation Errors | I've been feeling v tired lately, IDK why. | I've been feeling very tired lately, I don't know why. | ["I've been feeling very tired lately, I don't know why."] | grammar_correction.csv | test |
| 1760 | Abbreviation Errors | My flight was del'd, so I'll arr late. | My flight was delayed, so I'll arrive late. | ["My flight was delayed, so I'll arrive late."] | grammar_correction.csv | test |
| 1085 | Agreement in Comparative and Superlative Forms | He was more colder than his brother. | He was colder than his brother. | ['He was colder than his brother.'] | grammar_correction.csv | test |
| 1087 | Agreement in Comparative and Superlative Forms | She is most beautiful than her sister. | She is more beautiful than her sister. | ['She is more beautiful than her sister.'] | grammar_correction.csv | test |
| 1104 | Agreement in Comparative and Superlative Forms | He is the most strongest man in the competition. | He is the strongest man in the competition. | ['He is the strongest man in the competition.'] | grammar_correction.csv | test |
| 1114 | Agreement in Comparative and Superlative Forms | He is the most more skillful player on the team. | He is the most skillful player on the team. | ['He is the most skillful player on the team.'] | grammar_correction.csv | test |
| 1115 | Agreement in Comparative and Superlative Forms | The most shortest route is through the park. | The shortest route is through the park. | ['The shortest route is through the park.'] | grammar_correction.csv | test |

### `data/processed/grammar_correction_train.jsonl`

- Classification: `primary-dataset`
- File size: `496.6 KB`
- Examples: `1612`
- Input column: `original`
- Corrected column: `corrected`
- Label columns: `error_type`
- Metadata columns: `serial_number, source, split`

| column | dtype |
|---|---|
| serial_number | int64 |
| error_type | object |
| original | object |
| corrected | object |
| references | object |
| source | object |
| split | object |

| serial_number | error_type | original | corrected | references | source | split |
|---|---|---|---|---|---|---|
| 1720 | Abbreviation Errors | He works at FB, but I work at MS. | He works at Facebook, but I work at Microsoft. | ['He works at Facebook, but I work at Microsoft.'] | grammar_correction.csv | train |
| 1722 | Abbreviation Errors | I'll be there ASAP, pls wait. | I'll be there as soon as possible, please wait. | ["I'll be there as soon as possible, please wait."] | grammar_correction.csv | train |
| 1723 | Abbreviation Errors | She's a prof at UCLA, in the CS dept. | She's a professor at the University of California, Los Angeles, in the Computer Science department. | ["She's a professor at the University of California, Los Angeles, in the Computer Science department."] | grammar_correction.csv | train |
| 1724 | Abbreviation Errors | The engine has 200hp and 300lb-ft tq. | The engine has 200 horsepower and 300 pound-feet of torque. | ['The engine has 200 horsepower and 300 pound-feet of torque.'] | grammar_correction.csv | train |
| 1729 | Abbreviation Errors | Their business model involves B2B and B2C svs. | Their business model involves business-to-business and business-to-consumer services. | ['Their business model involves business-to-business and business-to-consumer services.'] | grammar_correction.csv | train |
| 1730 | Abbreviation Errors | The H2O lvl in the tank is getting low. | The water level in the tank is getting low. | ['The water level in the tank is getting low.'] | grammar_correction.csv | train |
| 1731 | Abbreviation Errors | I think they use AI and ML in their app. | I think they use artificial intelligence and machine learning in their app. | ['I think they use artificial intelligence and machine learning in their app.'] | grammar_correction.csv | train |
| 1732 | Abbreviation Errors | Plz send me the PDF, not the DOC file. | Please send me the Portable Document Format, not the Word Document file. | ['Please send me the Portable Document Format, not the Word Document file.'] | grammar_correction.csv | train |
| 1733 | Abbreviation Errors | They're going to visit the MOMA nxt wk. | They're going to visit the Museum of Modern Art next week. | ["They're going to visit the Museum of Modern Art next week."] | grammar_correction.csv | train |
| 1736 | Abbreviation Errors | My phone has GPS, but it's not v accurate. | My phone has Global Positioning System, but it's not very accurate. | ["My phone has Global Positioning System, but it's not very accurate."] | grammar_correction.csv | train |

### `data/processed/grammar_correction_validation.jsonl`

- Classification: `primary-dataset`
- File size: `62.8 KB`
- Examples: `204`
- Input column: `original`
- Corrected column: `corrected`
- Label columns: `error_type`
- Metadata columns: `serial_number, source, split`

| column | dtype |
|---|---|
| serial_number | int64 |
| error_type | object |
| original | object |
| corrected | object |
| references | object |
| source | object |
| split | object |

| serial_number | error_type | original | corrected | references | source | split |
|---|---|---|---|---|---|---|
| 1725 | Abbreviation Errors | In the meeting, they discussed GDPR and CCPA cmp. | In the meeting, they discussed General Data Protection Regulation and California Consumer Privacy Act compliance. | ['In the meeting, they discussed General Data Protection Regulation and California Consumer Privacy Act compliance.'] | grammar_correction.csv | validation |
| 1726 | Abbreviation Errors | My new TV has HDMI, USB, and BT cnct. | My new TV has High-Definition Multimedia Interface, Universal Serial Bus, and Bluetooth connectivity. | ['My new TV has High-Definition Multimedia Interface, Universal Serial Bus, and Bluetooth connectivity.'] | grammar_correction.csv | validation |
| 1728 | Abbreviation Errors | The pkg was del'd by USPS, not UPS. | The package was delivered by United States Postal Service, not United Parcel Service. | ['The package was delivered by United States Postal Service, not United Parcel Service.'] | grammar_correction.csv | validation |
| 1734 | Abbreviation Errors | My fav band is playing at SXSW this yr. | My favorite band is playing at South by Southwest this year. | ['My favorite band is playing at South by Southwest this year.'] | grammar_correction.csv | validation |
| 1754 | Abbreviation Errors | The team is focusing on R&D for the proj. | The team is focusing on research and development for the project. | ['The team is focusing on research and development for the project.'] | grammar_correction.csv | validation |
| 1098 | Agreement in Comparative and Superlative Forms | This recipe is most simpler than the other one. | This recipe is simpler than the other one. | ['This recipe is simpler than the other one.'] | grammar_correction.csv | validation |
| 1103 | Agreement in Comparative and Superlative Forms | The roller coaster was the most scariest I've ever been on. | The roller coaster was the scariest I've ever been on. | ["The roller coaster was the scariest I've ever been on."] | grammar_correction.csv | validation |
| 1116 | Agreement in Comparative and Superlative Forms | This is the most more affordable option for us. | This is the more affordable option for us. | ['This is the more affordable option for us.'] | grammar_correction.csv | validation |
| 1119 | Agreement in Comparative and Superlative Forms | The most easiest way to learn is by doing. | The easiest way to learn is by doing. | ['The easiest way to learn is by doing.'] | grammar_correction.csv | validation |
| 1120 | Agreement in Comparative and Superlative Forms | The most highest mountain in the world is Mount Everest. | The highest mountain in the world is Mount Everest. | ['The highest mountain in the world is Mount Everest.'] | grammar_correction.csv | validation |

### `data/prompt_registry.json`

- Classification: `support-metadata`
- File size: `1.4 KB`
- Examples: `3`
- Input column: `None`
- Corrected column: `None`
- Label columns: `none`
- Metadata columns: `version_id`

| column | dtype |
|---|---|
| version_id | str |
| template | str |
| description | str |
| created_at | str |
| metrics | dict |
| is_active | bool |

| version_id | template | description | created_at | metrics | is_active |
|---|---|---|---|---|---|
| v1.0.0 | Correct the following English sentence.
Sentence: {input}
Return only the corrected sentence. | Simple correction prompt with no external context. | 2026-05-25T00:00:00+00:00 | {'gleu': 0.61, 'exact_match': 0.52} | False |
| v1.1.0 | Use the grammar guidance below to correct the sentence.
Context:
{context}

Sentence: {input}
Return only the corrected sentence. | Context-augmented prompt for RAG-assisted correction. | 2026-05-25T00:00:01+00:00 | {'gleu': 0.68, 'exact_match': 0.58} | True |
| v2.0.0 | Identify the error type, reason briefly, and then provide the corrected sentence.
Context:
{context}

Sentence: {input} | Chain-of-thought style prompt with error type identification. | 2026-05-25T00:00:02+00:00 | {'gleu': 0.7, 'exact_match': 0.6} | False |

### `data/raw/grammar_correction.csv`

- Classification: `primary-dataset`
- File size: `248.8 KB`
- Examples: `2018`
- Input column: `Ungrammatical Statement`
- Corrected column: `Standard English`
- Label columns: `none`
- Metadata columns: `Serial Number`

| column | dtype |
|---|---|
| Serial Number | int64 |
| Error Type | object |
| Ungrammatical Statement | object |
| Standard English | object |

| Serial Number | Error Type | Ungrammatical Statement | Standard English |
|---|---|---|---|
| 1 | Verb Tense Errors | I goes to the store everyday. | I go to the store everyday. |
| 2 | Verb Tense Errors | They was playing soccer last night. | They were playing soccer last night. |
| 3 | Verb Tense Errors | She have completed her homework. | She has completed her homework. |
| 4 | Verb Tense Errors | He don't know the answer. | He doesn't know the answer. |
| 5 | Verb Tense Errors | The sun rise in the east. | The sun rises in the east. |
| 6 | Verb Tense Errors | I am eat pizza for lunch. | I am eating pizza for lunch. |
| 7 | Verb Tense Errors | The students studies for the exam. | The students study for the exam. |
| 8 | Verb Tense Errors | The car need to be repaired. | The car needs to be repaired. |
| 9 | Verb Tense Errors | She will goes to the party tonight. | She will go to the party tonight. |
| 10 | Verb Tense Errors | They watches the movie together. | They watch the movie together. |

### `data/raw/manifest.json`

- Classification: `support-metadata`
- File size: `414 B`
- Examples: `n/a`
- Input column: `None`
- Corrected column: `None`
- Label columns: `none`
- Metadata columns: `none`

| column | dtype |
|---|---|
| generated_at | str |
| files | list |

| generated_at | files |
|---|---|
| 2026-05-26T00:51:27.350978Z | [{'path': 'data/raw/grammar_correction.csv', 'size_bytes': 254751, 'sha256': '2dc565c3b993c0a1a04cacf2c73e35b03886cfb26771fb8fcb03a1df8542950a', 'rows': 2018, 'columns': ['Serial Number', 'Error Type', 'Ungrammatical Statement', 'Standard English']}] |

### `data/sample/grammar_rules.txt`

- Classification: `support-text`
- File size: `3.2 KB`
- Examples: `30`
- Input column: `None`
- Corrected column: `None`
- Label columns: `none`
- Metadata columns: `none`

| column | dtype |
|---|---|
| line | str |

| line |
|---|
| Subject-verb agreement rule 1: A singular subject normally takes a singular verb in the present tense. |
| Subject-verb agreement rule 2: A plural subject normally takes a plural verb in the present tense. |
| Subject-verb agreement rule 3: Collective nouns may take singular verbs when the group acts as one unit. |
| Subject-verb agreement rule 4: When subjects are joined by "and," use a plural verb unless the phrase names one idea. |
| Subject-verb agreement rule 5: When subjects are joined by "or" or "nor," match the verb to the nearer subject. |
| Subject-verb agreement rule 6: Indefinite pronouns such as everyone, each, and somebody usually take singular verbs. |
| Subject-verb agreement rule 7: Phrases between the subject and verb do not change the verb form. |
| Subject-verb agreement rule 8: Titles of books, movies, or organizations usually take singular verbs. |
| Subject-verb agreement rule 9: Amounts of time, money, or distance treated as a unit usually take singular verbs. |
| Subject-verb agreement rule 10: In there-is and there-are constructions, choose the verb based on the real subject that follows. |

### `data/sample/sample_sentences.txt`

- Classification: `support-text`
- File size: `1.9 KB`
- Examples: `20`
- Input column: `None`
- Corrected column: `None`
- Label columns: `none`
- Metadata columns: `none`

| column | dtype |
|---|---|
| line | str |

| line |
|---|
| She walk to school every morning. |
| The dogs barks at every stranger. |
| My brother and sister is coming tonight. |
| The list of items are on the desk. |
| Each student need a notebook for class. |
| He bought a umbrella before the storm. |
| She is best player on the team. |
| I saw an university near the station. |
| They adopted cat from shelter downtown. |
| We stayed at a old hotel by river. |

### `data/vector_store/chunks.json`

- Classification: `support-artifact`
- File size: `5.4 KB`
- Examples: `30`
- Input column: `None`
- Corrected column: `None`
- Label columns: `none`
- Metadata columns: `chunk_id, source`

| column | dtype |
|---|---|
| chunk_id | int |
| text | str |
| source | str |

| chunk_id | text | source |
|---|---|---|
| 0 | Subject-verb agreement rule 1: A singular subject normally takes a singular verb in the present tense. | document_0 |
| 1 | Subject-verb agreement rule 2: A plural subject normally takes a plural verb in the present tense. | document_1 |
| 2 | Subject-verb agreement rule 3: Collective nouns may take singular verbs when the group acts as one unit. | document_2 |
| 3 | Subject-verb agreement rule 4: When subjects are joined by "and," use a plural verb unless the phrase names one idea. | document_3 |
| 4 | Subject-verb agreement rule 5: When subjects are joined by "or" or "nor," match the verb to the nearer subject. | document_4 |
| 5 | Subject-verb agreement rule 6: Indefinite pronouns such as everyone, each, and somebody usually take singular verbs. | document_5 |
| 6 | Subject-verb agreement rule 7: Phrases between the subject and verb do not change the verb form. | document_6 |
| 7 | Subject-verb agreement rule 8: Titles of books, movies, or organizations usually take singular verbs. | document_7 |
| 8 | Subject-verb agreement rule 9: Amounts of time, money, or distance treated as a unit usually take singular verbs. | document_8 |
| 9 | Subject-verb agreement rule 10: In there-is and there-are constructions, choose the verb based on the real subject that follows. | document_9 |

### `data/vector_store/metadata.json`

- Classification: `support-artifact`
- File size: `186 B`
- Examples: `n/a`
- Input column: `None`
- Corrected column: `None`
- Label columns: `none`
- Metadata columns: `none`

| column | dtype |
|---|---|
| backend | str |
| embedding_model | str |
| chunk_size | int |
| chunk_overlap | int |
| chunk_count | int |
| vector_dimension | int |

| backend | embedding_model | chunk_size | chunk_overlap | chunk_count | vector_dimension |
|---|---|---|---|---|---|
| numpy | sentence-transformers/all-MiniLM-L6-v2 | 512 | 50 | 30 | 128 |


## Primary Dataset Schema

- Input column: `original`
- Corrected column: `corrected`
- Label columns: `error_type`
- Metadata columns: `serial_number, source`

### Column Types

| column | dtype |
|---|---|
| serial_number | int64 |
| error_type | object |
| original | object |
| corrected | object |
| references | object |
| source | object |

### Sample Rows

| serial_number | error_type | original | corrected | references | source |
|---|---|---|---|---|---|
| 1 | Verb Tense Errors | I goes to the store everyday. | I go to the store everyday. | ['I go to the store everyday.'] | grammar_correction.csv |
| 2 | Verb Tense Errors | They was playing soccer last night. | They were playing soccer last night. | ['They were playing soccer last night.'] | grammar_correction.csv |
| 3 | Verb Tense Errors | She have completed her homework. | She has completed her homework. | ['She has completed her homework.'] | grammar_correction.csv |
| 4 | Verb Tense Errors | He don't know the answer. | He doesn't know the answer. | ["He doesn't know the answer."] | grammar_correction.csv |
| 5 | Verb Tense Errors | The sun rise in the east. | The sun rises in the east. | ['The sun rises in the east.'] | grammar_correction.csv |
| 6 | Verb Tense Errors | I am eat pizza for lunch. | I am eating pizza for lunch. | ['I am eating pizza for lunch.'] | grammar_correction.csv |
| 7 | Verb Tense Errors | The students studies for the exam. | The students study for the exam. | ['The students study for the exam.'] | grammar_correction.csv |
| 8 | Verb Tense Errors | The car need to be repaired. | The car needs to be repaired. | ['The car needs to be repaired.'] | grammar_correction.csv |
| 9 | Verb Tense Errors | She will goes to the party tonight. | She will go to the party tonight. | ['She will go to the party tonight.'] | grammar_correction.csv |
| 10 | Verb Tense Errors | They watches the movie together. | They watch the movie together. | ['They watch the movie together.'] | grammar_correction.csv |

## Split Summary

| split | rows | percentage |
|---|---|---|
| train | 1612 | 79.88% |
| validation | 204 | 10.11% |
| test | 202 | 10.01% |

## Error Type Distribution

### Explicit Labels

| error_type | count | percentage |
|---|---|---|
| Sentence Structure Errors | 103 | 5.10% |
| Verb Tense Errors | 100 | 4.96% |
| Article Usage | 100 | 4.96% |
| Subject-Verb Agreement | 100 | 4.96% |
| Spelling Mistakes | 100 | 4.96% |
| Preposition Usage | 95 | 4.71% |
| Punctuation Errors | 60 | 2.97% |
| Relative Clause Errors | 51 | 2.53% |
| Ambiguity | 50 | 2.48% |
| Negation Errors | 50 | 2.48% |
| Tautology | 50 | 2.48% |
| Mixed Metaphors/Idioms | 50 | 2.48% |
| Incorrect Auxiliaries | 50 | 2.48% |
| Slang, Jargon, and Colloquialisms | 50 | 2.48% |
| Gerund and Participle Errors | 50 | 2.48% |
| Abbreviation Errors | 50 | 2.48% |
| Lack of Parallelism in Lists or Series | 50 | 2.48% |
| Agreement in Comparative and Superlative Forms | 49 | 2.43% |
| Passive Voice Overuse | 49 | 2.43% |
| Conjunction Misuse | 49 | 2.43% |
| Parallelism Errors | 49 | 2.43% |
| Faulty Comparisons | 49 | 2.43% |
| Inappropriate Register | 49 | 2.43% |
| Mixed Conditionals | 49 | 2.43% |
| Ellipsis Errors | 49 | 2.43% |
| Contractions Errors | 49 | 2.43% |
| Infinitive Errors | 49 | 2.43% |
| Quantifier Errors | 48 | 2.38% |
| Clichés | 48 | 2.38% |
| Pronoun Errors | 47 | 2.33% |
| Modifiers Misplacement | 46 | 2.28% |
| Capitalization Errors | 40 | 1.98% |
| Sentence Fragments | 40 | 1.98% |
| Word Choice/Usage | 40 | 1.98% |
| Run-on Sentences | 40 | 1.98% |
| Redundancy/Repetition | 20 | 0.99% |

### Normalized Audit Buckets

| bucket | count | percentage |
|---|---|---|
| mixed/multiple | 569 | 28.20% |
| preposition | 297 | 14.72% |
| article | 290 | 14.37% |
| spelling | 276 | 13.68% |
| tense | 248 | 12.29% |
| punctuation | 190 | 9.42% |
| subject-verb agreement | 108 | 5.35% |
| capitalization | 40 | 1.98% |

### Imbalance Report

- Largest explicit label: `Sentence Structure Errors` (5.10%)
- Smallest explicit label: `Redundancy/Repetition` (0.99%)
- Largest audit bucket: `mixed/multiple` (28.20%)
- Smallest audit bucket: `capitalization` (1.98%)

## Data Quality

- Average source sentence length: `9.019` tokens
- Average corrected sentence length: `8.762` tokens
- Vocabulary size: `3179`
- Type-token ratio: `0.1798`
- Empty original rows: `0`
- Empty corrected rows: `0`
- Suspicious label rows: `0`
- Noisy rows: `70`
- Exact pair duplicates: `3`

### Split Leakage

| comparison | original_overlap | pair_overlap |
|---|---|---|
| train_vs_validation | 1 | 1 |
| train_vs_test | 0 | 0 |
| validation_vs_test | 0 | 0 |

## BERT Token Imbalance Estimate

- Approximate source tokens: `18201`
- Approximate ERROR tokens: `3929`
- Approximate CORRECT tokens: `15291`
- Approximate ERROR token ratio: `0.2159`
- Approximate CORRECT:ERROR ratio: `3.892`

### Top Changed Tokens

| token | count |
|---|---|
| a | 194 |
| the | 189 |
| was | 130 |
| to | 93 |
| but | 88 |
| of | 77 |
| is | 75 |
| and | 75 |
| it | 73 |
| by | 60 |
| it's | 53 |
| with | 52 |
| in | 51 |
| most | 45 |
| on | 40 |

## Strengths

- The corpus already contains paired ungrammatical and corrected sentences, which is directly usable for sequence-to-sequence training.
- The dataset includes explicit high-level error_type labels, which makes targeted analysis and curriculum design possible.
- Train, validation, and test splits are present and sized sensibly for a 2,018-example corpus.
- Text cleanliness is high, with essentially no empty rows, control characters, or split leakage.

## Weaknesses

- The dataset is small for fine-tuning large grammar-correction models, especially for T5.
- Many examples belong to niche or stylistic categories rather than core grammar categories, which can dilute signal for sentence-level correction.
- Token-level ERROR labels are not explicitly provided, so BERT detection targets must be weakly inferred from diffs.

## Biggest Risks Before Training

- Overfitting is likely if T5 is trained only on this 2,018-row corpus without augmentation or external data.
- Weakly inferred token labels can make the BERT detector under-calibrated and class-imbalanced.
- Abbreviation, style, and structural rewrite examples may encourage broader rewriting instead of minimal grammar edits.

## Final Recommendations

- Oversampling needed: `True`
- Augmentation needed: `True`
- Use weighted cross-entropy for BERT token detection to offset the dominant CORRECT token class.
- Keep the current train/validation/test split, because no material pair leakage was detected.
- Consider oversampling core grammar categories such as article, punctuation, capitalization, and preposition errors if those are product priorities.
- Add external grammar-correction corpora such as JFLEG and CoNLL-2014 before serious T5 fine-tuning.
- Augment with synthetic perturbations or back-translation if you want stronger T5 generalization from this small corpus.

## Charts

- `error_type_distribution.svg`
- `sentence_length_distribution.svg`
- `top_repeated_tokens.svg`
- `duplicate_frequency.svg`
- `split_distribution.svg`
- `bias_indicators.svg`
