import csv

with open("test.csv", "r") as f:
    id_score_list = []
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        # print('line[{}] = {}'.format(i, line))
        score = 0
        answered = 0
        id = 0
        for response in line:
            ### @TODO look for ID value too!!
            # if found ID field: save it
            # eg id = response
            if response == "Not True":
                # score += 0
                answered += 1
            elif response == "Somewhat true":
                score += 1
                answered += 1
            elif response == "Very true":
                score += 2
                answered += 1
        avg_score = score / answered
        id_score_list.append((id, avg_score))
    
 # will then save into csv wh/ each line is all MT neurons for a "trial"
with open("./avg_scores.csv", 'w') as csv_f: 
    csv_w = csv.writer(csv_f)   
    csv_w.writerows(id_score_list)
