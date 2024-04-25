##
from scipy.stats import wasserstein_distance
import pandas as pd
import cv2
import matplotlib.image as mpimg
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from keras.preprocessing.image import ImageDataGenerator


seed = int(time.time())
print(seed)
random.seed(seed)
keras.utils.set_random_seed(seed)
##


def select_images_from_path(classe):
    PATH = f'./src/USPtex/images/{classe}'
    IMAGES = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".png"):
                IMAGES.append(os.path.join(root, file))
    print("Terminou a lista de imagens")
    return IMAGES


def add_noise(img, noise_level=0.6):
    if img.max() > 1:
        img = img / 255.0
    noise = np.random.normal(0, noise_level, img.shape)
    img_noisy = img + noise
    # Garante que os valores estejam entre 0 e 1
    img_noisy = np.clip(img_noisy, 0, 1)
    img_noisy = (img_noisy * 255).astype(np.uint8)
    return img_noisy


def exibir_imagens_grid(linhas, colunas, imagens):
    # Ajusta o tamanho do plot conforme necessário
    fig, axs = plt.subplots(linhas, colunas, figsize=(15, 15))
    # Transforma a matriz de eixos em um array unidimensional para facilitar o acesso

    for i in range(linhas * colunas):
        if i < len(imagens):
            img = mpimg.imread(imagens[i]) if isinstance(imagens[i], str) else imagens[
                i]  # Carrega a imagem se for um caminho de arquivo
            axs[i].imshow(img)
        axs[i].axis('off')  # Esconde os eixos para cada subplot

    plt.tight_layout()  # Ajusta o layout para minimizar sobreposições
    plt.show()


def imagens_para_vetores(lista_de_caminhos):
    vetores_de_imagens = []  # Lista para armazenar os arrays das imagens
    for caminho in lista_de_caminhos:
        imagem = Image.open(caminho)  # Carregar a imagem
        # Converter a imagem para um array numpy
        imagem_array = np.array(imagem)
        vetores_de_imagens.append(imagem_array)  # Adicionar o array na lista
    return vetores_de_imagens


def exibir_imagens_lado_a_lado(imagem1, imagem2, text, cmap='gray'):
    # Preparar para exibir as imagens
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(text, fontsize=16)
    # Exibir a imagem normal
    ax[0].imshow(imagem1, cmap=cmap)
    ax[0].axis('off')  # Esconder os eixos

    # Exibir a imagem latente ajustada
    ax[1].imshow(imagem2, cmap=cmap)
    ax[1].axis('off')  # Esconder os eixos

    fig.show()


def rgb_para_cinza(imagem_rgb):
    cinza = 0.2989 * imagem_rgb[:, :, 0] + 0.5870 * \
        imagem_rgb[:, :, 1] + 0.1140 * imagem_rgb[:, :, 2]
    return cinza.astype(np.uint8)


def calcular_emd(imagem1, imagem2):
    """
    Calcula o Earth Mover's Distance (EMD) entre duas imagens.

    Ambas as imagens devem ser arrays numpy em escala de cinza.
    """
    # Achatando as imagens para transformá-las em vetores 1D
    imagem1_cinza = cv2.cvtColor(
        imagem1, cv2.COLOR_BGR2GRAY) if imagem1.ndim == 3 else imagem1
    imagem2_cinza = cv2.cvtColor(
        imagem2, cv2.COLOR_BGR2GRAY) if imagem2.ndim == 3 else imagem2
    vetor1 = imagem1_cinza.flatten()
    vetor2 = imagem2_cinza.flatten()

    # Calculando o EMD
    emd = wasserstein_distance(vetor1, vetor2)

    return emd


def convert_img_latente(latent_img, decoder):
    latent_img = np.expand_dims(latent_img, axis=0)
    return decoder.predict(latent_img[0].reshape((-1, latentSize[0], latentSize[1], latentSize[2])))
# Função ajustada para processar e exibir imagens


def process_and_display_image(img, encoder, decoder, subplot_position, title, IMG_SIZE, is_latent=False, is_reconstructed=False):
    if not is_latent and not is_reconstructed:
        # Processamento para imagem original
        plt.subplot(2, 3, subplot_position)
        plt.title(title)
        plt.imshow(img.astype('uint8'), cmap='gray')
        plt.axis('off')
    elif is_latent:
        # Geração da representação latente
        # Ajuste para o formato esperado pelo modelo
        img_normalized = normalize_img(img).reshape((1, IMG_SIZE, IMG_SIZE, 3))
        latent_img = encoder.predict(img_normalized)
        mx, mn = np.max(latent_img), np.min(latent_img)
        latent_flat = ((latent_img - mn) * 255 / (mx - mn)).flatten(order='F')
        plt.subplot(2, 3, subplot_position)
        plt.title(title)
        plt.imshow(Image.fromarray(latent_flat[:2025].astype(
            'uint8').reshape((45, 45)), mode='L'))
        plt.axis('off')
        return latent_img  # Retornar para usar na reconstrução
    elif is_reconstructed:
        # Reconstrução da imagem a partir da representação latente
        decoded_imgs = decoder.predict(img.reshape((-1,) + latentSize))
        # Convertendo para tons de cinza pela média dos canais de cores
        img_reconstructed_gray = np.mean(decoded_imgs[0], axis=-1)
        plt.subplot(2, 3, subplot_position)
        plt.title(title)
        plt.imshow(img_reconstructed_gray, cmap='gray')
        plt.axis('off')

# Função para normalizar as imagens


def normalize_img(img):
    return img / 255.0


def reconstruct_image(img, encoder, decoder, IMG_SIZE):
    # Convertendo imagem em tons de cinza para RGB "falso"
    # Verifica se a imagem é em tons de cinza
    if len(img.shape) == 2 or img.shape[2] == 1:
        img_rgb = np.stack((img,) * 3, axis=-1)
    else:
        img_rgb = img

    # Normalizando a imagem
    img_normalized = normalize_img(img_rgb).reshape((1, IMG_SIZE, IMG_SIZE, 3))

    # Obtendo a representação latente
    latent_img = encoder.predict(img_normalized)

    # Reconstruindo a imagem a partir da representação latente
    decoded_imgs = decoder.predict(latent_img)

    # Convertendo para tons de cinza pela média dos canais de cores
    img_reconstructed_gray = np.mean(decoded_imgs[0], axis=-1)

    # Retornando a imagem reconstruída
    return img_reconstructed_gray
##


def experimento_1(classe, noise_level, plot=False):

    IMAGES = select_images_from_path(classe)
    images = imagens_para_vetores(IMAGES)
    # seleciona uma imagem aleatória

    img_teste_ori = random.choice(images)
    img_teste_noisy = add_noise(img_teste_ori, noise_level)
    # calc emd

    img_teste_ori_gray = cv2.cvtColor(img_teste_ori, cv2.COLOR_BGR2GRAY)
    img_teste_noisy_gray = cv2.cvtColor(
        add_noise(img_teste_noisy, noise_level), cv2.COLOR_BGR2GRAY)

    # Preparando o plot
    plt.figure(figsize=(15, 10))
    # Processando e exibindo a imagem original, latente, e reconstruída para img_teste_ori
    latent_img_teste_ori = process_and_display_image(
        img_teste_ori_gray, encoder, decoder, 1, 'Original', IMG_SIZE)
    latent_img_teste_ori = process_and_display_image(
        img_teste_ori, encoder, decoder, 2, 'Latent', IMG_SIZE, is_latent=True)
    process_and_display_image(latent_img_teste_ori, encoder, decoder,
                              3, 'Reconstructed', IMG_SIZE, is_reconstructed=True)

    # Processando e exibindo a imagem original, latente, e reconstruída para img_teste_noisy
    latent_img_teste_noisy = process_and_display_image(
        img_teste_noisy_gray, encoder, decoder, 4, 'Original', IMG_SIZE)
    latent_img_teste_noisy = process_and_display_image(
        img_teste_noisy, encoder, decoder, 5, 'Latent', IMG_SIZE, is_latent=True)
    process_and_display_image(latent_img_teste_noisy, encoder,
                              decoder, 6, 'Reconstructed', IMG_SIZE, is_reconstructed=True)

    reconstructed_img_teste_ori = reconstruct_image(
        img_teste_ori_gray, encoder, decoder, IMG_SIZE)
    reconstructed_img_teste_noisy = reconstruct_image(
        img_teste_noisy_gray, encoder, decoder, IMG_SIZE)
    if plot:

        plt.savefig('experimento1.png')
        plt.show()
    # calcular emd para imagens originais (Com e sem ruído)
    emd_ori = calcular_emd(img_teste_ori_gray, img_teste_noisy_gray)

    # calcular emd para imagens reconstruídas (Com e sem ruído)
    emd_reconstructed = calcular_emd(
        reconstructed_img_teste_ori, reconstructed_img_teste_noisy)
    # print('Noise level:', noise_level)
    # print(f'EMD para imagens originais: {emd_ori}')
    # print(f'EMD para imagens reconstruídas: {emd_reconstructed}')
    plt.close()
    return emd_ori, emd_reconstructed


##


def experimento_2(PATH, imgens_p_geracao=3, imagens_geradas=3):

    IMAGES = []  # Aqui deve ser uma lista com os nomes dos arquivos das imagens

    # Gerar a lista de todos os arquivos de imagem nas 10 pastas
    IMAGES = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            # Assumindo que as imagens são .jpg; ajuste conforme necessário
            if file.endswith(".png"):
                IMAGES.append(os.path.join(root, file))
    print("Terminou a lista de imagens")

    selected_images_paths = np.random.choice(
        IMAGES, imgens_p_geracao, replace=False)
    print('Imagens selecionadas:', selected_images_paths)
    images = np.zeros((imgens_p_geracao, 128, 128, 3))
    print("Selecionou as imagens")

    for i, img_path in enumerate(selected_images_paths):
        img = Image.open(img_path)
        # Ajustar o redimensionamento para garantir que a imagem seja cortada para 128x128
        img = img.resize(
            (128, int(img.size[1] / (img.size[0] / 128))), Image.ANTIALIAS)
        # Ajuste no corte para obter uma imagem centralizada de 128x128
        # Este passo assume que o redimensionamento anterior mantém a proporção correta
        if img.size[1] > 128:
            img = img.crop(
                (0, (img.size[1] - 128) / 2, 128, (img.size[1] + 128) / 2))
        elif img.size[1] < 128:
            img = img.crop(
                ((img.size[0] - 128) / 2, 0, (img.size[0] + 128) / 2, 128))
        images[i, :, :, :] = np.asarray(img).astype('float32') / 255.
    print("Carregou as imagens")

    # Calcular elipsoide a partir das 100 imagens selecionadas
    encoded_imgs = encoder.predict(images)
    sz = latentSize[0] * latentSize[1] * latentSize[2]
    encoded_imgs = encoded_imgs.reshape((-1, sz))
    mm = np.mean(encoded_imgs, axis=0)
    ss = np.cov(encoded_imgs, rowvar=False)
    print("Calculou a elipsoide")

    # generated = np.median(encoded_imgs,axis=0)

    # Gerar 9 imagens aleatórias de textura
    generated = np.random.multivariate_normal(mm, ss, imagens_geradas)
    generated = generated.reshape(
        (-1, latentSize[0], latentSize[1], latentSize[2]))
    print("Gerou as imagens - 1")

    plt.figure(figsize=(imagens_geradas * 5, 5))
    plt.title(f'Imagens geradas - Media e Covariancia -{imgens_p_geracao}')
    for k in range(imagens_geradas):
        # Gerar as imagens decodificadas a partir das representações latentes
        plt.title(f'Imagens geradas - Media e Covariancia -{imgens_p_geracao}')
        decoded_imgs = decoder.predict(generated[k].reshape(
            (-1, latentSize[0], latentSize[1], latentSize[2])))
        img = Image.fromarray(
            (255 * decoded_imgs[0]).astype('uint8').reshape((128, 128, 3)))

        # Adicionar um subplot para cada imagem na posição correta

        plt.subplot(1, imagens_geradas, k+1)
        plt.imshow(img)
        plt.axis('off')

    # Mostrar o plot com todas as imagens

    plt.tight_layout()  # Ajusta automaticamente os subplots para que caibam na figura
    plt.show()
    plt.savefig(f'./img/experimento2-{imgens_p_geracao}-media.png')
    plt.close()

    print("Exibiu as imagens")

    mm = np.median(encoded_imgs, axis=0)
    generated = np.random.multivariate_normal(mm, ss, imagens_geradas)
    generated = generated.reshape(
        (-1, latentSize[0], latentSize[1], latentSize[2]))
    print("Gerou as imagens - 2")

    plt.figure(figsize=(imagens_geradas * 5, 5))
    plt.title(f'Imagens geradas - Mediana e Covariancia -{imgens_p_geracao}')
    for k in range(imagens_geradas):
        plt.title(
            f'Imagens geradas - Mediana e Covariancia -{imgens_p_geracao}')
        # Gerar as imagens decodificadas a partir das representações latentes
        decoded_imgs = decoder.predict(generated[k].reshape(
            (-1, latentSize[0], latentSize[1], latentSize[2])))
        img = Image.fromarray(
            (255 * decoded_imgs[0]).astype('uint8').reshape((128, 128, 3)))

        # Adicionar um subplot para cada imagem na posição correta

        plt.subplot(1, imagens_geradas, k+1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'./img/experimento2-{imgens_p_geracao}-mediana.png')

##


def experimento_2_beta(PATH, imgens_p_geracao=3, imagens_geradas=3, beta=0.35):

    IMAGES = []  # Aqui deve ser uma lista com os nomes dos arquivos das imagens

    # Gerar a lista de todos os arquivos de imagem nas 10 pastas
    IMAGES = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            # Assumindo que as imagens são .jpg; ajuste conforme necessário
            if file.endswith(".png"):
                IMAGES.append(os.path.join(root, file))
    print("Terminou a lista de imagens")

    selected_images_paths = np.random.choice(
        IMAGES, imgens_p_geracao, replace=False)
    print('Imagens selecionadas:', selected_images_paths)
    images = np.zeros((imgens_p_geracao, 128, 128, 3))
    print("Selecionou as imagens")

    for i, img_path in enumerate(selected_images_paths):
        img = Image.open(img_path)
        # Ajustar o redimensionamento para garantir que a imagem seja cortada para 128x128
        img = img.resize(
            (128, int(img.size[1] / (img.size[0] / 128))), Image.ANTIALIAS)
        # Ajuste no corte para obter uma imagem centralizada de 128x128
        # Este passo assume que o redimensionamento anterior mantém a proporção correta
        if img.size[1] > 128:
            img = img.crop(
                (0, (img.size[1] - 128) / 2, 128, (img.size[1] + 128) / 2))
        elif img.size[1] < 128:
            img = img.crop(
                ((img.size[0] - 128) / 2, 0, (img.size[0] + 128) / 2, 128))
        images[i, :, :, :] = np.asarray(img).astype('float32') / 255.
    print("Carregou as imagens")

    # Calcular elipsoide a partir das 100 imagens selecionadas
    encoded_imgs = encoder.predict(images)
    sz = latentSize[0] * latentSize[1] * latentSize[2]
    encoded_imgs = encoded_imgs.reshape((-1, sz))
    mm = np.mean(encoded_imgs, axis=0)
    ss = np.cov(encoded_imgs, rowvar=False)
    print("Calculou a elipsoide")

    # generated = np.median(encoded_imgs,axis=0)

    generated = np.random.multivariate_normal(mm, ss, imagens_geradas)
    generated = beta * generated + (1 - beta) * encoded_imgs[:imagens_geradas]
    print("Gerou as imagens - 1")

    plt.figure(figsize=(imagens_geradas * 5, 5))
    # plt.title(f'Imagens geradas - Media e Covariancia -{imgens_p_geracao}')
    for k in range(imagens_geradas):
        # Gerar as imagens decodificadas a partir das representações latentes
        # plt.title(f'Imagens geradas - Media e Covariancia -{imgens_p_geracao}')
        decoded_imgs = decoder.predict(generated[k].reshape(
            (-1, latentSize[0], latentSize[1], latentSize[2])))
        img = Image.fromarray(
            (255 * decoded_imgs[0]).astype('uint8').reshape((128, 128, 3)))

        # Adicionar um subplot para cada imagem na posição correta

        plt.subplot(1, imagens_geradas, k+1)
        plt.imshow(img)
        plt.axis('off')

    # Mostrar o plot com todas as imagens

    plt.tight_layout()  # Ajusta automaticamente os subplots para que caibam na figura

    plt.savefig(
        f'./img/experimento2_beta_{int(beta*100)}-{imgens_p_geracao}-media.png')
    plt.show()
    plt.close()

    print("Exibiu as imagens")

    mm = np.median(encoded_imgs, axis=0)
    generated = np.random.multivariate_normal(mm, ss, imagens_geradas)
    generated = beta * generated + (1 - beta) * encoded_imgs[:imagens_geradas]
    print("Gerou as imagens - 2")

    plt.figure(figsize=(imagens_geradas * 5, 5))
    # plt.title(f'Imagens geradas - Mediana e Covariancia -{imgens_p_geracao}')
    for k in range(imagens_geradas):
        # plt.title(f'Imagens geradas - Mediana e Covariancia -{imgens_p_geracao}')
        # Gerar as imagens decodificadas a partir das representações latentes
        decoded_imgs = decoder.predict(generated[k].reshape(
            (-1, latentSize[0], latentSize[1], latentSize[2])))
        img = Image.fromarray(
            (255 * decoded_imgs[0]).astype('uint8').reshape((128, 128, 3)))

        # Adicionar um subplot para cada imagem na posição correta

        plt.subplot(1, imagens_geradas, k+1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(
        f'./img/experimento2_beta_{int(beta*100)}-{imgens_p_geracao}-mediana.png')
    plt.show()

##


def load_images_and_encode(folder, encoder):
    encoded_features = []
    labels = []
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    image = Image.open(image_path).convert(
                        'RGB')  # Garante que a imagem está em RGB
                    # Redimensiona a imagem conforme o input do autoencoder
                    image = image.resize((128, 128))
                    image_array = np.array(image)
                    image_array = image_array / 255.0
                    image_array = np.expand_dims(image_array, axis=0)
                    latent_representation = encoder.predict(image_array)
                    encoded_image = latent_representation.flatten()
                    encoded_features.append(encoded_image)
                    labels.append(class_name)
                except Exception as e:
                    print(
                        f"Erro ao carregar ou codificar a imagem {image_path}: {e}")
    return np.array(encoded_features), labels


def apply_pca(encoded_features, n_components=2):
    pca = PCA(n_components=n_components)
    transformed_features = pca.fit_transform(encoded_features)
    return transformed_features


def plot_images(transformed_features, labels, mean_point):
    plt.figure(figsize=(12, 8))  # Aumenta o tamanho da figura
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(transformed_features[indices, 0],
                    transformed_features[indices, 1], label=label)

    # Plotando o ponto médio
    plt.scatter(mean_point[0], mean_point[1], color='red',
                marker='X', s=100, label='Ponto Médio')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
    plt.title('Vetores latentes em 2D após PCA')

    plt.tight_layout(pad=1)
    plt.savefig('pca_with_mean_point.png')
    plt.show()

##


def experimento_2_path_completo(PATH, imagens_p_geracao=3, imagens_geradas=3):
    IMAGES = []  # Lista para armazenar os caminhos das imagens encontradas

    # Percorre o diretório especificado e seus subdiretórios para encontrar arquivos .png
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".png"):
                IMAGES.append(os.path.join(root, file))
    print("Terminou a lista de imagens")

    # Ajustar imagens_p_geracao baseado no número de imagens disponíveis
    if len(IMAGES) < imagens_p_geracao:
        print(
            f"Apenas {len(IMAGES)} imagens encontradas. Ajustando imgens_p_geracao para {len(IMAGES)}.")
        imagens_p_geracao = len(IMAGES)

    if imagens_p_geracao > 0:
        # Selecionar aleatoriamente um subconjunto das imagens encontradas
        selected_images_paths = np.random.choice(
            IMAGES, imagens_p_geracao, replace=False)
        print('Imagens selecionadas:', selected_images_paths)

        # Preparar um array para armazenar as imagens carregadas e processadas
        images = np.zeros((imagens_p_geracao, 128, 128, 3))

        for i, img_path in enumerate(selected_images_paths):
            # Converter para RGB se necessário
            img = Image.open(img_path).convert('RGB')
            # Redimensionar mantendo a proporção
            img.thumbnail((128, 128), Image.ANTIALIAS)
            # Criar uma nova imagem com fundo branco para garantir que a imagem final seja 128x128
            final_img = Image.new('RGB', (128, 128), (255, 255, 255))
            # Calcular posição para colar a imagem redimensionada
            upper_left = ((128 - img.width) // 2, (128 - img.height) // 2)
            final_img.paste(img, upper_left)
            images[i, :, :, :] = np.asarray(final_img).astype('float32') / 255.
        print("Carregou as imagens")
    else:
        print("Nenhuma imagem .png encontrada.")
        return

    # Calcular elipsoide a partir das 100 imagens selecionadas
    encoded_imgs = encoder.predict(images)
    sz = latentSize[0] * latentSize[1] * latentSize[2]
    encoded_imgs = encoded_imgs.reshape((-1, sz))
    mm = np.mean(encoded_imgs, axis=0)
    ss = np.cov(encoded_imgs, rowvar=False)
    print("Calculou a elipsoide")

    # generated = np.median(encoded_imgs,axis=0)

    # Gerar 9 imagens aleatórias de textura
    generated = np.random.multivariate_normal(mm, ss, imagens_geradas)
    generated = generated.reshape(
        (-1, latentSize[0], latentSize[1], latentSize[2]))
    print("Gerou as imagens - 1")

    plt.figure(figsize=(imagens_geradas * 5, 5))
    plt.title(f'Imagens geradas - Media e Covariancia -{imagens_p_geracao}')
    for k in range(imagens_geradas):
        # Gerar as imagens decodificadas a partir das representações latentes
        plt.title(
            f'Imagens geradas - Media e Covariancia -{imagens_p_geracao}')
        decoded_imgs = decoder.predict(generated[k].reshape(
            (-1, latentSize[0], latentSize[1], latentSize[2])))
        img = Image.fromarray(
            (255 * decoded_imgs[0]).astype('uint8').reshape((128, 128, 3)))

        # Adicionar um subplot para cada imagem na posição correta

        plt.subplot(1, imagens_geradas, k+1)
        plt.imshow(img)
        plt.axis('off')

    # Mostrar o plot com todas as imagens

    plt.tight_layout()  # Ajusta automaticamente os subplots para que caibam na figura
    plt.show()
    plt.savefig(
        f'./img/completo/experimento2_completo-{imagens_p_geracao}-media.png')
    plt.close()

    print("Exibiu as imagens")

    mm = np.median(encoded_imgs, axis=0)
    generated = np.random.multivariate_normal(mm, ss, imagens_geradas)
    generated = generated.reshape(
        (-1, latentSize[0], latentSize[1], latentSize[2]))
    print("Gerou as imagens - 2")

    plt.figure(figsize=(imagens_geradas * 5, 5))
    plt.title(f'Imagens geradas - Mediana e Covariancia -{imagens_p_geracao}')
    for k in range(imagens_geradas):
        plt.title(
            f'Imagens geradas - Mediana e Covariancia -{imagens_p_geracao}')
        # Gerar as imagens decodificadas a partir das representações latentes
        decoded_imgs = decoder.predict(generated[k].reshape(
            (-1, latentSize[0], latentSize[1], latentSize[2])))
        img = Image.fromarray(
            (255 * decoded_imgs[0]).astype('uint8').reshape((128, 128, 3)))

        # Adicionar um subplot para cada imagem na posição correta

        plt.subplot(1, imagens_geradas, k+1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(
        f'./img/completo/experimento2_completo-{imagens_p_geracao}-mediana.png')


def experimento_2_beta_simples(PATH, imgens_p_geracao=3, imagens_geradas=3, beta=0.35):

    IMAGES = []  # Aqui deve ser uma lista com os nomes dos arquivos das imagens

    # Gerar a lista de todos os arquivos de imagem nas 10 pastas
    IMAGES = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            # Assumindo que as imagens são .jpg; ajuste conforme necessário
            if file.endswith(".png"):
                IMAGES.append(os.path.join(root, file))
    print("Terminou a lista de imagens")

    selected_images_paths = np.random.choice(
        IMAGES, imgens_p_geracao, replace=False)
    print('Imagens selecionadas:', selected_images_paths)
    images = np.zeros((imgens_p_geracao, 128, 128, 3))
    print("Selecionou as imagens")

    for i, img_path in enumerate(selected_images_paths):
        img = Image.open(img_path)
        # Ajustar o redimensionamento para garantir que a imagem seja cortada para 128x128
        img = img.resize(
            (128, int(img.size[1] / (img.size[0] / 128))), Image.ANTIALIAS)
        # Ajuste no corte para obter uma imagem centralizada de 128x128
        # Este passo assume que o redimensionamento anterior mantém a proporção correta
        if img.size[1] > 128:
            img = img.crop(
                (0, (img.size[1] - 128) / 2, 128, (img.size[1] + 128) / 2))
        elif img.size[1] < 128:
            img = img.crop(
                ((img.size[0] - 128) / 2, 0, (img.size[0] + 128) / 2, 128))
        images[i, :, :, :] = np.asarray(img).astype('float32') / 255.
    print("Carregou as imagens")

    # Calcular elipsoide a partir das 100 imagens selecionadas
    encoded_imgs = encoder.predict(images)
    sz = latentSize[0] * latentSize[1] * latentSize[2]
    encoded_imgs = encoded_imgs.reshape((-1, sz))
    mm = np.mean(encoded_imgs, axis=0)
    ss = np.cov(encoded_imgs, rowvar=False)
    print("Calculou a elipsoide")

    # generated = np.median(encoded_imgs,axis=0)

    generated = np.random.multivariate_normal(mm, ss, imagens_geradas)
    generated = beta * generated + (1 - beta) * encoded_imgs[:imagens_geradas]
    print("Gerou as imagens - 1")

    plt.figure(figsize=(imagens_geradas * 5, 5))
    # plt.title(f'Imagens geradas - Media e Covariancia -{imgens_p_geracao}')
    for k in range(imagens_geradas):
        # Gerar as imagens decodificadas a partir das representações latentes
        # plt.title(f'Imagens geradas - Media e Covariancia -{imgens_p_geracao}')
        decoded_imgs = decoder.predict(generated[k].reshape(
            (-1, latentSize[0], latentSize[1], latentSize[2])))
        img = Image.fromarray(
            (255 * decoded_imgs[0]).astype('uint8').reshape((128, 128, 3)))

        # Adicionar um subplot para cada imagem na posição correta

        plt.subplot(1, imagens_geradas, k+1)
        plt.imshow(img)
        plt.axis('off')

    # Mostrar o plot com todas as imagens

    plt.tight_layout()  # Ajusta automaticamente os subplots para que caibam na figura

    plt.savefig(
        f'./img/experimento2_beta_{int(beta*100)}-{imgens_p_geracao}-media.png')
    plt.show()
    plt.close()

    print("Exibiu as imagens")


def add_salt_and_pepper_noise(img, salt_pepper_ratio=0.5, amount=0.1):
    """
    Adiciona ruído sal e pimenta a uma imagem.

    Parâmetros:
        img (ndarray): Imagem de entrada. Deve estar em escala de 0 a 255.
        salt_pepper_ratio (float): Proporção de sal (branco) para pimenta (preto). 0.5 significa mesmo número de sal e pimenta.
        amount (float): Porcentagem de pixels que terão ruído adicionado.

    Retorna:
        ndarray: Imagem com ruído adicionado.
    """
    # Se a imagem estiver em escala de 0 a 1, converta para 0 a 255
    if img.max() <= 1:
        img = img * 255

    # Cria uma cópia da imagem original para adicionar ruído
    noisy_img = np.copy(img)

    # Calcula o número de pixels afetados pelo ruído
    num_salt = np.ceil(amount * img.size * salt_pepper_ratio)
    num_pepper = np.ceil(amount * img.size * (1.0 - salt_pepper_ratio))

    # Adiciona ruído Sal (branco)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[tuple(coords)] = 255

    # Adiciona ruído Pimenta (preto)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[tuple(coords)] = 0

    # Converte de volta para o tipo de dados original e escala
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return noisy_img


def experimento_1_sal_e_pimeta(classe, noise_level, plot=False):

    IMAGES = select_images_from_path(classe)
    images = imagens_para_vetores(IMAGES)
    # seleciona uma imagem aleatória

    img_teste_ori = random.choice(images)
    img_teste_noisy = add_salt_and_pepper_noise(
        img_teste_ori, amount=noise_level)
    # calc emd

    img_teste_ori_gray = cv2.cvtColor(img_teste_ori, cv2.COLOR_BGR2GRAY)
    img_teste_noisy_gray = cv2.cvtColor(img_teste_noisy, cv2.COLOR_BGR2GRAY)

    # Preparando o plot
    plt.figure(figsize=(15, 10))
    # Processando e exibindo a imagem original, latente, e reconstruída para img_teste_ori
    latent_img_teste_ori = process_and_display_image(
        img_teste_ori_gray, encoder, decoder, 1, 'Original', IMG_SIZE)
    latent_img_teste_ori = process_and_display_image(
        img_teste_ori, encoder, decoder, 2, 'Latent', IMG_SIZE, is_latent=True)
    process_and_display_image(latent_img_teste_ori, encoder, decoder,
                              3, 'Reconstructed', IMG_SIZE, is_reconstructed=True)

    # Processando e exibindo a imagem original, latente, e reconstruída para img_teste_noisy
    latent_img_teste_noisy = process_and_display_image(
        img_teste_noisy_gray, encoder, decoder, 4, 'Original', IMG_SIZE)
    latent_img_teste_noisy = process_and_display_image(
        img_teste_noisy, encoder, decoder, 5, 'Latent', IMG_SIZE, is_latent=True)
    process_and_display_image(latent_img_teste_noisy, encoder,
                              decoder, 6, 'Reconstructed', IMG_SIZE, is_reconstructed=True)

    reconstructed_img_teste_ori = reconstruct_image(
        img_teste_ori_gray, encoder, decoder, IMG_SIZE)
    reconstructed_img_teste_noisy = reconstruct_image(
        img_teste_noisy_gray, encoder, decoder, IMG_SIZE)
    if plot:

        plt.savefig(f'./img/sal_e_pimenta/{classe}_{int(ruido*100)}')
        plt.show()
    # calcular emd para imagens originais (Com e sem ruído)
    emd_ori = calcular_emd(img_teste_ori_gray, img_teste_noisy_gray)

    # calcular emd para imagens reconstruídas (Com e sem ruído)
    emd_reconstructed = calcular_emd(
        reconstructed_img_teste_ori, reconstructed_img_teste_noisy)
    # print('Noise level:', noise_level)
    # print(f'EMD para imagens originais: {emd_ori}')
    # print(f'EMD para imagens reconstruídas: {emd_reconstructed}')
    plt.close()
    return emd_ori, emd_reconstructed


def add_shot_noise(img, scale=1):
    """
    Adiciona ruído de shot (Poisson) a uma imagem.

    Parâmetros:
        img (ndarray): Imagem de entrada. Deve estar em escala de 0 a 255.
        scale (float): Fator de escala para ajustar a intensidade do ruído.

    Retorna:
        ndarray: Imagem com ruído adicionado.
    """
    # Converte a imagem para float para evitar problemas de tipo de dados
    img_float = img.astype(np.float32)

    # O ruído de Poisson é proporcional à imagem. A escala permite ajustar a intensidade do ruído.
    if scale != 1:
        img_float = img_float * scale

    # Gera o ruído de Poisson, onde a variância é igual à média
    noisy_img = np.random.poisson(img_float).astype(np.float32)

    # Reescala para manter a mesma escala da imagem original
    if scale != 1:
        noisy_img = noisy_img / scale

    # Garante que a imagem resultante ainda esteja dentro dos limites de 0 a 255 e converta de volta para uint8
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return noisy_img


def experimento_1_poisson(classe, noise_level, plot=False):

    IMAGES = select_images_from_path(classe)
    images = imagens_para_vetores(IMAGES)
    # seleciona uma imagem aleatória

    img_teste_ori = random.choice(images)
    img_teste_noisy = add_shot_noise(img_teste_ori, scale=noise_level)
    # calc emd

    img_teste_ori_gray = cv2.cvtColor(img_teste_ori, cv2.COLOR_BGR2GRAY)
    img_teste_noisy_gray = cv2.cvtColor(img_teste_noisy, cv2.COLOR_BGR2GRAY)

    # Preparando o plot
    plt.figure(figsize=(15, 10))
    # Processando e exibindo a imagem original, latente, e reconstruída para img_teste_ori
    latent_img_teste_ori = process_and_display_image(
        img_teste_ori_gray, encoder, decoder, 1, 'Original', IMG_SIZE)
    latent_img_teste_ori = process_and_display_image(
        img_teste_ori, encoder, decoder, 2, 'Latent', IMG_SIZE, is_latent=True)
    process_and_display_image(latent_img_teste_ori, encoder, decoder,
                              3, 'Reconstructed', IMG_SIZE, is_reconstructed=True)

    # Processando e exibindo a imagem original, latente, e reconstruída para img_teste_noisy
    latent_img_teste_noisy = process_and_display_image(
        img_teste_noisy_gray, encoder, decoder, 4, 'Original', IMG_SIZE)
    latent_img_teste_noisy = process_and_display_image(
        img_teste_noisy, encoder, decoder, 5, 'Latent', IMG_SIZE, is_latent=True)
    process_and_display_image(latent_img_teste_noisy, encoder,
                              decoder, 6, 'Reconstructed', IMG_SIZE, is_reconstructed=True)

    reconstructed_img_teste_ori = reconstruct_image(
        img_teste_ori_gray, encoder, decoder, IMG_SIZE)
    reconstructed_img_teste_noisy = reconstruct_image(
        img_teste_noisy_gray, encoder, decoder, IMG_SIZE)
    if plot:

        plt.savefig(f'./img/poisson/{classe}_{int(ruido*100)}')
        plt.show()
    # calcular emd para imagens originais (Com e sem ruído)
    emd_ori = calcular_emd(img_teste_ori_gray, img_teste_noisy_gray)

    # calcular emd para imagens reconstruídas (Com e sem ruído)
    emd_reconstructed = calcular_emd(
        reconstructed_img_teste_ori, reconstructed_img_teste_noisy)
    # print('Noise level:', noise_level)
    # print(f'EMD para imagens originais: {emd_ori}')
    # print(f'EMD para imagens reconstruídas: {emd_reconstructed}')
    plt.close()
    return emd_ori, emd_reconstructed


def cria_imagens_com_toda_base(porcento, encoder, decoder, img_geradas=9):
    IMAGES = []
    for root, dirs, files in os.walk('./src/USPtex/images'):
        for file in files:
            # Assumindo que as imagens são .jpg; ajuste conforme necessário
            if file.endswith(".png"):
                IMAGES.append(os.path.join(root, file))
    print("Terminou a lista de imagens")
    n_imagens = int(len(IMAGES) * porcento)

    selected_images_paths = np.random.choice(IMAGES, n_imagens, replace=False)
    images = np.zeros((100, 128, 128, 3))
    print("Selecionou as imagens")

    for i, img_path in enumerate(selected_images_paths):
        img = Image.open(img_path)
        # Ajustar o redimensionamento para garantir que a imagem seja cortada para 128x128
        img = img.resize(
            (128, int(img.size[1] / (img.size[0] / 128))), Image.ANTIALIAS)
        # Ajuste no corte para obter uma imagem centralizada de 128x128
        # Este passo assume que o redimensionamento anterior mantém a proporção correta
        if img.size[1] > 128:
            img = img.crop(
                (0, (img.size[1] - 128) / 2, 128, (img.size[1] + 128) / 2))
        elif img.size[1] < 128:
            img = img.crop(
                ((img.size[0] - 128) / 2, 0, (img.size[0] + 128) / 2, 128))
        images[i, :, :, :] = np.asarray(img).astype('float32') / 255.

    # Calcular elipsoide a partir das 100 imagens selecionadas
    encoded_imgs = encoder.predict(images)
    sz = latentSize[0] * latentSize[1] * latentSize[2]
    encoded_imgs = encoded_imgs.reshape((-1, sz))
    mm = np.mean(encoded_imgs, axis=0)
    ss = np.cov(encoded_imgs, rowvar=False)

    # Gerar 9 imagens aleatórias de textura
    generated = np.random.multivariate_normal(mm, ss, 9)
    generated = generated.reshape(
        (-1, latentSize[0], latentSize[1], latentSize[2]))

    # Exibir as 9 imagens aleatórias
    for k in range(int(img_geradas/3)):
        plt.figure(figsize=(15, 5))
        for j in range(1, 4):
            decoded_imgs = decoder.predict(
                generated[k * 3 + j - 1].reshape((-1, latentSize[0], latentSize[1], latentSize[2])))
            img = Image.fromarray(
                (255 * decoded_imgs[0]).astype('uint8').reshape((128, 128, 3)))
            plt.subplot(1, 3, j)
            plt.imshow(img)
            plt.axis('off')
        plt.savefig(
            f'./img/00-gerada_bd_completa_{int(porcento*100)}_k{k}.png')
        plt.show()
##


if __name__ == '__main__':
    BATCH_SIZE = 12
    EPOCHS = 30
    IMG_SIZE = 128
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_batches = train_datagen.flow_from_directory('./src/USPtex/images',
                                                      target_size=(IMG_SIZE, IMG_SIZE), shuffle=True, class_mode='input',
                                                      batch_size=BATCH_SIZE)

    # ENCODER
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = Conv2D(48, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    encoded = Conv2D(32, (1, 1), activation='relu', padding='same')(x)

    # LATENT SPACE
    latentSize = (16, 16, 32)

    # DECODER
    direct_input = Input(shape=latentSize)
    x = Conv2D(192, (1, 1), activation='relu', padding='same')(direct_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # COMPILE
    encoder = Model(input_img, encoded)
    decoder = Model(direct_input, decoded)
    autoencoder = Model(input_img, decoder(encoded))

    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    history = autoencoder.fit(train_batches,
                              steps_per_epoch=train_batches.samples // BATCH_SIZE,
                              epochs=EPOCHS, verbose=2)

    # Experimento 1 -  Gaussiano
    ruidos = [0.1, 0.3, 0.5, 0.9]
    lst_ori_final = []
    lst_recon_final = []
    for ruido in ruidos:
        ori_lst = []
        recon_lst = []
        for i in range(1, 100):
            print(f'Experimento {i} - Ruido {ruido}')
            emd_ori, emd_reconstructed = experimento_1('c149', ruido)
            ori_lst.append(emd_ori)
            recon_lst.append(emd_reconstructed)

        media_ori = np.mean(ori_lst)
        media_recon = np.mean(recon_lst)
        lst_ori_final.append(media_ori)
        lst_recon_final.append(media_recon)

    for i in range(len(ruidos)):
        print(f'Ruído {ruidos[i]}')
        print('Média EMD para imagens originais:', lst_ori_final[i])
        print('Média EMD para imagens reconstruídas:', lst_recon_final[i])

    experimento_1('c149', 0.3, plot=True)

    # Experimento 1 - sal e pimenta
    classes = ['c003', 'c051', 'c065', 'c072', 'c081',
               'c089', 'c113', 'c136', 'c139', 'c149']  # classe = 'c149'
    for classe in classes:
        ruidos = [0.1, 0.3, 0.5, 0.9]
        lst_ori_final = []
        lst_recon_final = []

        for ruido in ruidos:
            plot = False
            ori_lst = []
            recon_lst = []
            for i in range(1, 100):
                print(f'Experimento {i} - Ruido {ruido} - {classe}')
                if i == 99:
                    plot = True
                emd_ori, emd_reconstructed = experimento_1_sal_e_pimeta(
                    classe, ruido, plot)
                ori_lst.append(emd_ori)
                recon_lst.append(emd_reconstructed)

            media_ori = np.mean(ori_lst)
            media_recon = np.mean(recon_lst)
            lst_ori_final.append(media_ori)
            lst_recon_final.append(media_recon)

        df_resultados = pd.DataFrame({
            'Nível de Ruído': ruidos,
            'Média EMD Originais': lst_ori_final,
            'Média EMD Reconstruídas': lst_recon_final
        })
        # print(df_resultados)
        df_resultados.to_excel(
            f'./img/sal_e_pimenta/results/resultados_emd_sal_e_pimenta_{classe}.xlsx', index=False)
    # Experimento 1 - Poisson

    for classe in classes:
        ruidos = [0.1, 0.3, 0.5, 0.9]
        lst_ori_final = []
        lst_recon_final = []

        for ruido in ruidos:
            plot = False
            ori_lst = []
            recon_lst = []
            for i in range(1, 100):
                print(f'Experimento {i} - Ruido {ruido} - {classe}')
                if i == 99:
                    plot = True
                emd_ori, emd_reconstructed = experimento_1_poisson(
                    classe, ruido, plot)
                ori_lst.append(emd_ori)
                recon_lst.append(emd_reconstructed)

            media_ori = np.mean(ori_lst)
            media_recon = np.mean(recon_lst)
            lst_ori_final.append(media_ori)
            lst_recon_final.append(media_recon)

        df_resultados = pd.DataFrame({
            'Nível de Ruído': ruidos,
            'Média EMD Originais': lst_ori_final,
            'Média EMD Reconstruídas': lst_recon_final
        })
        # print(df_resultados)
    df_resultados.to_excel(
        f'./img/poisson/results/resultados_poisson_{classe}.xlsx', index=False)

    # Experimento 2

    PATH = './src/USPtex/images/c081'
    experimento_2(PATH, imgens_p_geracao=3, imagens_geradas=3)
    experimento_2(PATH, imgens_p_geracao=6, imagens_geradas=3)
    experimento_2(PATH, imgens_p_geracao=9, imagens_geradas=3)
    experimento_2(PATH, imgens_p_geracao=12, imagens_geradas=3)

    # Experimento 2 Com todos os arquivos
    PATH = './src/USPtex/images/'
    n = int(120*0.3)
    experimento_2_path_completo(PATH, imagens_p_geracao=n, imagens_geradas=3)
    n = int(120*0.6)
    experimento_2_path_completo(PATH, imagens_p_geracao=n, imagens_geradas=3)
    n = int(120*0.9)
    experimento_2_path_completo(PATH, imagens_p_geracao=n, imagens_geradas=3)

    # Experimento 2 - Beta
    PATH = './src/USPtex/images/'
    experimento_2_beta(PATH, imgens_p_geracao=3, imagens_geradas=3, beta=0.25)
    experimento_2_beta(PATH, imgens_p_geracao=6, imagens_geradas=3, beta=0.25)

    experimento_2_beta(PATH, imgens_p_geracao=3, imagens_geradas=3, beta=0.35)
    experimento_2_beta(PATH, imgens_p_geracao=6, imagens_geradas=3, beta=0.35)

    experimento_2_beta(PATH, imgens_p_geracao=3, imagens_geradas=3, beta=0.45)
    experimento_2_beta(PATH, imgens_p_geracao=6, imagens_geradas=3, beta=0.45)

    # Experimento 2 - Beta Simples (somente uma classe)
    PATH = './src/USPtex/images/c149'
    experimento_2_beta_simples(PATH, imgens_p_geracao=3,
                               imagens_geradas=3, beta=-0.55)
    experimento_2_beta_simples(PATH, imgens_p_geracao=3,
                               imagens_geradas=3, beta=-0.65)

    # Experimento 2 - Beta Simples (todas as classes)
    cria_imagens_com_toda_base(0.3, encoder, decoder)
    cria_imagens_com_toda_base(0.5, encoder, decoder)
    cria_imagens_com_toda_base(0.7, encoder, decoder)
