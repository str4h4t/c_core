import correlation_engine as ce
import pickle


if __name__ == "__main__":
    print("Starting Procedure...")
    f = open("config.txt","r")
    config = []
    for x in f:
        config.append(x.split('"')[1])
    ip_file = config[0]
    model = config[1]
    parameters = config[2]
    data_format = config[3]
    corr = ce.correlation_engine(ip_file, model, data_format)
    result = corr.execute()

    print("Ending Procedure...")