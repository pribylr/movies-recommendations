from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from process_data import load_and_process_data, id_valid, recalculate_recommendations, \
    find_names_of_movies, load_recommendations, find_not_rated_movies, add_rating, change_user_flag

app = Flask(__name__)
app.secret_key = 'idk'

movies = None
ratings = None
current_similarity = 'pearson'


@app.route('/')
def home():
    #session.clear()
    global movies, ratings
    movies, ratings = load_and_process_data()
    return render_template('index.html')


@app.route('/signin')
def recommend():
    return render_template('signin.html')


@app.route('/process-id', methods=['GET', 'POST'])
def process_id():
    user_id = request.form.get('user_id') or session.get('user_id')
    if user_id and id_valid(user_id=user_id, data=ratings):
        session['user_id'] = user_id
        if user_id == '0':
            return render_template('admin.html')
        else:
            user_id = int(user_id)
            user_ratings = find_names_of_movies(user_id)  # dict: movie name -> rating
            recommendations = load_recommendations(current_similarity)
            recommendations = recommendations[user_id]
            not_rated_movies = find_not_rated_movies(user_id)
            return render_template('user.html', user_id=user_id, user_ratings=user_ratings,
                                   recommendations=recommendations, not_rated_movies=not_rated_movies)
    else:
        flash('User ID not found')
        return render_template('signin.html', show_choices=False)


@app.route('/save-selection', methods=['POST'])
def save_selection():
    data = request.json
    try:
        print('what did i rated', data['selected_movie'], data['selected_rating'], data['user_id'])
        add_rating(int(data['user_id']), data['selected_movie'], int(data['selected_rating']))
        change_user_flag(int(data['user_id']))
        return jsonify({'status': 'success', 'message': 'Selection saved successfully'})
    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'message': 'An error occurred while saving the selection'}), 500


@app.route('/admin/rec')
def admin_rec():
    return render_template('admin.html', show_choices=True)


@app.route('/process-choice', methods=['POST'])
def process_choice():
    similarity_choice = request.form['similarity']
    how_many_users = request.form['how_many_users']
    global current_similarity
    current_similarity = similarity_choice
    res = recalculate_recommendations(similarity_choice, int(how_many_users))

    return render_template('result.html', similarity=similarity_choice, how_many_users=how_many_users)


@app.route('/signout')
def signout():
    #session.clear()
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
