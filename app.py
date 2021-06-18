from flask import Flask, render_template, request
import main as m

app = Flask(__name__)  # creating flask app


@app.route("/", methods=["GET", "POST"])  # to create front interface
def front():
    # if request.method == "POST":
    # final = main.KeyWordExtractor(z, no_of_keywords)
    return render_template("front_end.html")


@app.route("/sub", methods=["GET", "POST"])  # to create submit interface
def submit():
    if request.method == "POST":
        try:
            query = request.form['userquery']  # gets query of user
            no_of_keywords = request.form['keywords']  # gets number of queries required by user
            youtube = m.YoutubeLinkGenerator(query)  # calling youtube link function from main.py
            google_links = m.SearchableLinkGenerator(query)  # calling google link function from main.py
            head_n_sum = m.HeadlineAndSummaryGenerator(google_links,
                                                       query)  # calling headline and summary function from main.py
            text_list_feed = m.TextListFeeder(head_n_sum[1])  # calling text feed function from main.py
            final_keywords = m.KeyWordExtractor(text_list_feed,
                                                int(no_of_keywords))  # calling keyword generating function from main.py
            output = m.Output(google_links, head_n_sum, final_keywords)  # calling output function from main.py
            return render_template("sub.html", o1=output[0][0], o2=output[1][0], o3=output[2][0], o4=output[3][0],
                                   o5=output[0][1], o6=output[1][1], o7=output[2][1], o8=output[3][1], o9=output[0][2],
                                   o10=output[1][2], o11=output[2][2], o12=output[3][2], o13=output[0][3],
                                   o14=output[1][3], o15=output[2][3], o16=output[3][3], o17=output[0][4],
                                   o18=output[1][4], o19=output[2][4], o20=output[3][4],
                                   o21=youtube)  # rendering outputs to web interface
        except:
            return render_template("sub.html", o1=output[0][0], o2=output[1][0], o3=output[2][0], o4=output[3][0],
                                   o5=output[0][1], o6=output[1][1], o7=output[2][1], o8=output[3][1], o9=output[0][2],
                                   o10=output[1][2], o11=output[2][2], o12=output[3][2], o13=output[0][3],
                                   o14=output[1][3], o15=output[2][3], o16=output[3][3],
                                   o21=youtube)  # rendering outputs to web interface


if __name__ == "__main__":
    app.run(debug=True)
