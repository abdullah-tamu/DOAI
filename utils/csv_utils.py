import pandas as pd
import os

def write_results(results:dict, cur_class, total_classes, csv_path):
    keys = list(results.keys())

    if not os.path.exists(csv_path):
        df_all = None
        for class_name in total_classes:
            r = dict()
            for k in keys:
                r[k] = 0.00
            df_temp = pd.DataFrame(r, index=[class_name])

            if df_all is None:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all, df_temp], axis=0)

        df_all.to_csv(csv_path, header=True, float_format='%.2f')

    df = pd.read_csv(csv_path, index_col=0)

    for k in keys:
        df.loc[cur_class, k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')

def save_metric(metrics, total_classes, class_name, dataset, csv_path):
    results = dict()

    results[f'i_roc'] = metrics['i_roc']
    results[f'p_roc'] = metrics['p_roc']
    results[f'p_pro'] = metrics['p_pro']

    if dataset != 'mvtec':
        for indx in range(len(total_classes)):
            total_classes[indx] = f"{dataset}-{total_classes[indx]}"
        class_name = f"{dataset}-{class_name}"
    write_results(results, class_name, total_classes, csv_path)


def write_res_csv(res, res_path,append=False):
    import csv
    if append:
        mode='a'
    else:
        mode='w'
    with open(res_path + '/results.csv', mode, newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # for i in res:
        #     spamwriter.writerow([str(i[0]), str(i[1])])
        spamwriter.writerow([str(res[0]), str(res[1])])



