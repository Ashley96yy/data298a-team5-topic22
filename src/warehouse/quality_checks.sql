-- Each query returns a count of violations. Zero is required.
-- check: interaction_required_fields
SELECT count(*) FROM interactions WHERE user_idx IS NULL OR item_idx IS NULL
 OR timestamp IS NULL OR pos IS NULL OR row_id IS NULL OR split IS NULL
 OR split NOT IN ('train', 'val', 'test');

-- check: unique_user_item_pairs
SELECT count(*) FROM (SELECT user_idx, item_idx FROM interactions
 GROUP BY user_idx, item_idx HAVING count(*) <> 1);

-- check: unique_source_rows
SELECT count(*) FROM (SELECT row_id FROM interactions GROUP BY row_id HAVING count(*) <> 1);

-- check: interaction_foreign_keys
SELECT count(*) FROM interactions i LEFT JOIN user_map u USING(user_idx)
 LEFT JOIN item_map p USING(item_idx) WHERE u.user_idx IS NULL OR p.item_idx IS NULL;

-- check: leave_one_out_split
SELECT count(*) FROM (SELECT user_idx FROM interactions GROUP BY user_idx
 HAVING count(*) FILTER (WHERE split='val') <> 1
 OR count(*) FILTER (WHERE split='test') <> 1
 OR count(*) FILTER (WHERE split='train') < 1);

-- check: chronological_positions_and_splits
WITH ranked AS (SELECT *, row_number() OVER
 (PARTITION BY user_idx ORDER BY timestamp, row_id)-1 expected_pos,
 count(*) OVER (PARTITION BY user_idx) total FROM interactions)
SELECT count(*) FROM ranked WHERE pos <> expected_pos OR split <>
 CASE WHEN expected_pos=total-1 THEN 'test' WHEN expected_pos=total-2 THEN 'val' ELSE 'train' END;

-- check: sequence_lengths
SELECT count(*) FROM sequences WHERE train_items IS NULL OR train_ts IS NULL
 OR n_train IS NULL OR n_train <> len(train_items) OR n_train <> len(train_ts)
 OR val_item IS NULL OR test_item IS NULL OR val_ts IS NULL OR test_ts IS NULL;

-- check: sequence_interaction_agreement
WITH differences AS (
 (SELECT user_idx,item_idx,timestamp,pos,split FROM sequence_events
 EXCEPT ALL SELECT user_idx,item_idx,timestamp,pos,split FROM interactions)
 UNION ALL
 (SELECT user_idx,item_idx,timestamp,pos,split FROM interactions
 EXCEPT ALL SELECT user_idx,item_idx,timestamp,pos,split FROM sequence_events))
SELECT count(*) FROM differences;

-- check: evaluation_target_agreement
WITH differences AS (
 (SELECT user_idx,target,split FROM eval_cases
 EXCEPT ALL SELECT user_idx,item_idx,split FROM interactions WHERE split IN ('val','test'))
 UNION ALL
 (SELECT user_idx,item_idx,split FROM interactions WHERE split IN ('val','test')
 EXCEPT ALL SELECT user_idx,target,split FROM eval_cases))
SELECT count(*) FROM differences;

-- check: item_map_catalog_agreement
WITH differences AS (
 (SELECT item_idx,parent_asin FROM items EXCEPT ALL SELECT item_idx,parent_asin FROM item_map)
 UNION ALL
 (SELECT item_idx,parent_asin FROM item_map EXCEPT ALL SELECT item_idx,parent_asin FROM items))
SELECT count(*) FROM differences;

-- check: item_text_catalog_agreement
WITH differences AS (
 (SELECT item_idx,parent_asin FROM item_text EXCEPT ALL SELECT item_idx,parent_asin FROM item_map)
 UNION ALL
 (SELECT item_idx,parent_asin FROM item_map EXCEPT ALL SELECT item_idx,parent_asin FROM item_text))
SELECT count(*) FROM differences;

-- check: item_titles_present
SELECT count(*) FROM items WHERE title IS NULL OR trim(title)='';

-- check: item_text_present
SELECT count(*) FROM item_text WHERE item_text IS NULL OR trim(item_text)='';

-- check: training_review_provenance
SELECT count(*) FROM train_reviews r LEFT JOIN user_map u USING(user_id)
 LEFT JOIN item_map p USING(parent_asin)
 WHERE r.split IS DISTINCT FROM 'train' OR r.item_idx IS DISTINCT FROM p.item_idx
 OR NOT EXISTS (SELECT 1 FROM interactions i WHERE i.user_idx=u.user_idx
 AND i.item_idx=r.item_idx AND i.split='train');

-- check: user_group_train_counts
SELECT count(*) FROM user_groups g FULL OUTER JOIN sequences s USING(user_idx)
 WHERE g.user_idx IS NULL OR s.user_idx IS NULL OR g.n_train IS DISTINCT FROM s.n_train;

-- check: item_group_train_counts
WITH counts AS (SELECT item_idx,count(*) n FROM interactions WHERE split='train' GROUP BY item_idx)
SELECT count(*) FROM item_groups g FULL OUTER JOIN item_map m USING(item_idx)
 LEFT JOIN counts c ON c.item_idx=m.item_idx
 WHERE g.item_idx IS NULL OR m.item_idx IS NULL OR g.train_count IS DISTINCT FROM coalesce(c.n,0);

-- check: client_train_counts
SELECT count(*) FROM clients c FULL OUTER JOIN sequences s ON c.client_id=s.user_idx
 WHERE c.client_id IS NULL OR s.user_idx IS NULL
 OR c.n_local_train IS DISTINCT FROM s.n_train OR c.n_local_total IS DISTINCT FROM s.n_train+2;
