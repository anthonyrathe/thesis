from os import path
experiment = "super_experiment_3"
expert = "no_second_layer_not_peer_group_aware"

no_problem = True
for str1 in ["StackedEnv","StackedEnvDiff","StackedEnvBinary"]:
    for str2 in ["0.005", "0.0002","1-11"]:
        filename = "{}_{}_{}".format(experiment_name, str1, str2)
        for i in range(100)

            if not path.exists("../experiment_results/{}/logs/{}_log_0_group_{}.csv".format(experiment,filename,str(i))):
                print("ERROR")
                no_problem = False

            if not path.exists("../experiment_results/{}/saved_models/{}_{}.model".format(experiment,filename,str(i))):
                print("ERROR")
                no_problem = False

        if not path.exists("../experiment_results/{}/saved_models/{}.model".format(experiment, filename)):
            print("ERROR")
            no_problem = False

if no_problem:
    print("OK")


