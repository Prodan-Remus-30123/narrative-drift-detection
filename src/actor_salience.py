from entities import analyze_entities


def compute_actor_salience(grouped_texts):
    salience_results = {}

    for period, texts in grouped_texts.items():
        entity_stats = analyze_entities(texts)
        salience_results[period] = {}

        for entity, stats in entity_stats.items():
            verb_count = sum(stats["verbs"].values())
            subject_count = stats["subject_count"]
            object_count = stats["object_count"]

            salience = (verb_count + subject_count + object_count)

            if salience > 0:
                salience_results[period][entity] = salience

    return salience_results

def compute_total_actor_salience(salience_results):
    totals = {}

    for period, entities in salience_results.items():
        for entity, score in entities.items():
            totals[entity] = (totals.get(entity, 0) + score)

    return totals