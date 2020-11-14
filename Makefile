default:
	python -m dnnseg.bin.grid_search eng.ini exps_conll20.yml -o conll20
	python -m dnnseg.bin.grid_search eng_baseline.ini exps_conll20.yml -o conll20
	python -m dnnseg.bin.grid_search xit.ini exps_conll20.yml -o conll20
	python -m dnnseg.bin.grid_search xit_baseline.ini exps_conll20.yml -o conll20
	rm conll20/*wb0_wf0*ini

pbs:
	python -m dnnseg.bin.make_jobs conll20/*.ini

gather_metrics:
	python -m dnnseg.bin.gather_metrics ../results/dnnseg/conll20/eng/ -m ".*eval_table_eng.*" -o eval_table_eng2eng.csv
	python -m dnnseg.bin.gather_metrics ../results/dnnseg/conll20/eng/ -m ".*eval_table_xit.*" -o eval_table_eng2xit.csv
	python -m dnnseg.bin.gather_metrics ../results/dnnseg/conll20/xit/ -m ".*eval_table_eng.*" -o eval_table_xit2eng.csv
	python -m dnnseg.bin.gather_metrics ../results/dnnseg/conll20/xit/ -m ".*eval_table_xit.*" -o eval_table_xit2xit.csv
	python -m dnnseg.bin.gather_metrics ../results/dnnseg/conll20/eng_baseline/ -m ".*eval_table_eng.*" -o eval_table_engbase2eng.csv
	python -m dnnseg.bin.gather_metrics ../results/dnnseg/conll20/eng_baseline/ -m ".*eval_table_xit.*" -o eval_table_engbase2xit.csv
	python -m dnnseg.bin.gather_metrics ../results/dnnseg/conll20/xit_baseline/ -m ".*eval_table_eng.*" -o eval_table_xitbase2eng.csv
	python -m dnnseg.bin.gather_metrics ../results/dnnseg/conll20/xit_baseline/ -m ".*eval_table_xit.*" -o eval_table_xitbase2xit.csv
