#include <cstdint>
#include <random>
#include <cmath>
#include <string>
#include <utility>
#include <fstream>
#include <iostream>

#ifdef WIN32
#define DLLEXPORT __declspec(dllexport)
#endif


extern "C" {

DLLEXPORT int getRandomIntValue(int max) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(0, max);

    int randomValue = distribution(generator);
    return randomValue;
}

DLLEXPORT float get_random_float_value(float max) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(0, max);

    float randomValue = distribution(generator);
    return randomValue;
}


DLLEXPORT int
get_label_two_output(int index, int fooball_img_list_size, int basket_img_list_size, const std::string& className) {

    std::string class_img;
    if (index >= 0 && index <= fooball_img_list_size - 1) {
        class_img = "football";
    } else if (index >= fooball_img_list_size && index <= fooball_img_list_size + basket_img_list_size - 1) {
        class_img = "basket";
    } else {
        class_img = "tennis";
    }
    if (class_img == className) {
        return 1;
    } else {
        return 0;
    }
}


DLLEXPORT int *set_label_three_output(int fooball_img_list_size, int basket_img_list_size, int tennis_img_list_size) {
    int *label_list = new int[fooball_img_list_size + basket_img_list_size + tennis_img_list_size];
    for (int i = 0; i < fooball_img_list_size; i++) {
        label_list[i] = 1;
    }
    for (int i = fooball_img_list_size; i < basket_img_list_size + fooball_img_list_size; i++) {
        label_list[i] = 2;
    }
    for (int i = basket_img_list_size + fooball_img_list_size;
         i < tennis_img_list_size + basket_img_list_size + fooball_img_list_size; i++) {
        label_list[i] = 3;
    }
    return label_list;
}


DLLEXPORT float sigmoid(float output) {
    //std::cout << output << std::endl;
    //std::cout <<output << std::endl;
    float exp = expf(output);
    //std::cout <<"exp : " << exp << std::endl;
    //std::cout << "sigmoid : " << 1 / (1 + (1 / exp)) << std::endl;
    return 1 / (1 + (1 / exp));
}
// Fonction d'activation qui retourne l'output du modèle binaire
DLLEXPORT int get_class_two_output(float output) {
    if (sigmoid(output) >= 0.7) {
        return 1;
    } else {
        return 0;
    }

}

DLLEXPORT float normalize_input(float input, float min_value, float max_value) {
    //std::cout << "normalize_niput " << (input - min_value) / (max_value - min_value) << std::endl;
    return (input - min_value) / (max_value - min_value);
}

DLLEXPORT float get_weighted_sum(const int *input, const float *weight, int img_size) {
    float weighted_sum = 0;
    for (int i = 0; i < img_size; i++) {
        //std::cout << "input " << input[i] << " weight " << weight[i] << std::endl;
        weighted_sum += normalize_input(input[i], 0.0, 255.0) * weight[i];
    }
    //std::cout << "weighted_sum " << weighted_sum << std::endl;
    return weighted_sum + 1;
}
//Interroge le modèle à 3 sortie et retourne la classe prédite
DLLEXPORT int
get_class_three_output(int *img, float *weight_football, float *weight_basket, float *weight_tennis, int img_size) {
    auto *output = new float[3];
    output[0] = sigmoid(get_weighted_sum(img, weight_football, img_size));
    output[1] = sigmoid(get_weighted_sum(img, weight_basket, img_size));
    output[2] = sigmoid(get_weighted_sum(img, weight_tennis, img_size));
    std::cout << "Football" << output[0] << std::endl;
    std::cout << "Basket" << output[1] << std::endl;
    std::cout << "Tennis" << output[2] << std::endl;

    int index_biggest_output = 0;
    for (int i = 1; i < 3; i++) {
        if (output[i] > output[index_biggest_output]) {
            index_biggest_output = i;
        }
    }
    delete[] output;
    return index_biggest_output + 1;
}

DLLEXPORT float *initialize_weight(int size) {
    auto *weight_list = new float[size];
    for (int i = 0; i < size; i++) {
        weight_list[i] = get_random_float_value(0.0000002);
        //std::cout << "poid : " << weight_list[i] << std::endl;
    }
    return weight_list;
}

/* Interroge le modèle,
   Met à jour les poids
   Et retourne 1 si le modèle trouve la bonne réponse sinon retourne 0*/
DLLEXPORT int get_output_and_set_weight(int *input, float *weight, int weight_list_size, int label) {
    float weighted_sum = get_weighted_sum(input, weight, weight_list_size);
    int output = get_class_two_output(weighted_sum);
    for (int i = 0; i < weight_list_size; i++) {
        weight[i] = weight[i] + 0.00001 * (label - output) * input[i];
    }
    return output;
}

DLLEXPORT void save_modele(float *weight_list, int list_size, const std::string& path) {
    std::ofstream outputFile(path);
    if (!outputFile) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << path << std::endl;
        return;
    }
    for (int i = 0; i < list_size; i++) {
        outputFile << weight_list[i] << std::endl;
    }
    outputFile.close();
    //std::cout << "File written successfully." << std::endl;
}

DLLEXPORT float *load_model(int nb_weight, const std::string &path) {
    auto *weight_list = new float[nb_weight];
    std::ifstream inputFile(path);
    if (!inputFile) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << path << std::endl;
        return nullptr;
    }

    //std::cout << "nb weight" << nb_weight << std::endl;

    for (int i = 0; i < nb_weight; i++) {
        if (!(inputFile >> weight_list[i])) {
            std::cerr << "Erreur de lecture du fichier." << std::endl;
            std::cerr << "poids erreur : " << i << std::endl;
            delete[] weight_list;
            return nullptr;
        }
        //std::cout << "index poids" << i << std::endl;
    }
    return weight_list;

}


DLLEXPORT int **
test_linear_model(int **img_list, const int *nb_image_per_class, int img_size, const char *football_model_path,
                  const char *basket_model_path, const char *tennis_model_path) {
    int **label_and_output = new int *[2];
    // Chargement des 3 modèle binaire entrainé
    float *football_model = load_model(img_size, football_model_path);
    float *basket_model = load_model(img_size, basket_model_path);
    float *tennis_model = load_model(img_size, tennis_model_path);

    int fooball_img_list_size = nb_image_per_class[0];
    int basket_img_list_size = nb_image_per_class[1];
    int tennis_img_list_size = nb_image_per_class[2];
    int img_list_size = fooball_img_list_size + basket_img_list_size + tennis_img_list_size;

    // Liste montrant les classe prédit par le modèle
    // 1 : football, 2 basket, 3 : tennis
    int *output_list = new int[img_list_size];

    // On interroge le modèle pour tout les images du dataset de test
    for (int i = 0; i < img_list_size; i++) {
        output_list[i] = get_class_three_output(img_list[i], football_model, basket_model, tennis_model, img_size);
    }


    label_and_output[0] = set_label_three_output(fooball_img_list_size, basket_img_list_size, tennis_img_list_size);
    label_and_output[1] = output_list;
    //std::cout << "fonction terminé" << std::endl;
    return label_and_output;
}

/* Fonction qui entraine un modèle binaire,
 enregistre le modèle dans un fichier,
et retourne une liste montrant l'évolution de la réussite du modèle au fil des itérations */
DLLEXPORT int *
train_linear_model(int **img_list, const int *nb_image_per_class, int img_size, int training_iteration,
                   const std::string &file_path, const std::string &className) {
    int nb_fooball_img_list = nb_image_per_class[0];
    int nb_basket_img_list = nb_image_per_class[1];
    int nb_img_list_size = nb_image_per_class[2];
    int img_list_size = nb_fooball_img_list + nb_basket_img_list + nb_img_list_size;

    //Initilisation des poids à des valeurs aléatoires
    float *weight_list = initialize_weight(img_size);

    //Liste montrant l'évolution du modèle au fil des itérations (à but statistiques)
    //1 si le modèle trouve la bonne réponse sinon 0
    auto *output_list = new int[training_iteration];

    int random_img_index;
    int label;
    int output;


    for (int i = 0; i < training_iteration; i++) {
        //Tirage d'une image au hasard
        random_img_index = getRandomIntValue(img_list_size - 1);

        //Affectation du label
        label = get_label_two_output(random_img_index, nb_fooball_img_list, nb_basket_img_list, className);

        //Mise à jour des poids
        output = get_output_and_set_weight(img_list[random_img_index], weight_list, img_size, label);

        //si le modèle à trouver la bonne réponse output_list[i] = 1  sinon output_list[i] = 0
        if (output == label) {
            output_list[i] = 1;
        } else {
            output_list[i] = 0;
        }
        std::cout << i << " / " << training_iteration << std::endl;
    }
    save_modele(weight_list, img_size, file_path);

    return output_list;
}
//Charge les 3 modèle bianire entrainé et retourne la classe prédite par le modèle à 3 sortie
DLLEXPORT int predict_class(int *img, int img_size, const char *football_model_path, const char *basket_model_path,
                            const char *tennis_model_path) {
    float *football_model = load_model(img_size, football_model_path);
    float *basket_model = load_model(img_size, basket_model_path);
    float *tennis_model = load_model(img_size, tennis_model_path);
    return get_class_three_output(img, football_model, basket_model, tennis_model, img_size);
    //return 1;
}

DLLEXPORT int *
train_linear_model_football(int **img_list, const int *size_class, int img_size, int training_iteration,
                            const char *file_path) {
    return train_linear_model(img_list, size_class, img_size, training_iteration, file_path, "football");
}

DLLEXPORT int *train_linear_model_basket(int **img_list, const int *size_class, int img_size, int training_iteration,
                                            const char *file_path) {
    return train_linear_model(img_list, size_class, img_size, training_iteration, file_path, "basket");
}

DLLEXPORT int *train_linear_model_tennis(int **img_list, const int *size_class, int img_size, int training_iteration,
                                            const char *file_path) {
    return train_linear_model(img_list, size_class, img_size, training_iteration, file_path, "tennis");
}


}