// central.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define SIZE 4

int A[SIZE][SIZE] = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 8, 7, 6},
    {4, 3, 2, 1}
};

int B[SIZE][SIZE] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}
};

int C[SIZE][SIZE];

void send_to_worker(char* ip, int port, int row_index) {
    struct sockaddr_in serv_addr;
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("Socket creation"); exit(1); }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &serv_addr.sin_addr);

    connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

    // Send 1 row of A and full B
    send(sock, A[row_index], sizeof(A[row_index]), 0);
    send(sock, B, sizeof(B), 0);

    // Receive 1 row of result
    recv(sock, C[row_index], sizeof(C[row_index]), 0);

    close(sock);
}

int main() {
    int ports[4] = {5001, 5002, 5003, 5004};

    for (int i = 0; i < 4; i++) {
        send_to_worker("worker" + (i+1), ports[i], i);
    }

    printf("Combined matrix:\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }
    return 0;
}
