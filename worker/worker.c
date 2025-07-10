// worker.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define SIZE 4

int main(int argc, char *argv[]) {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int row_count;

    int A_part[SIZE];       // 1 row of A
    int B[SIZE][SIZE];      // full matrix B
    int result[SIZE];       // one row result

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    int port = atoi(argv[1]);
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    bind(server_fd, (struct sockaddr *)&address, sizeof(address));
    listen(server_fd, 1);

    printf("Worker listening on port %d...\n", port);

    new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);

    // Receive A_part and matrix B
    recv(new_socket, A_part, sizeof(A_part), 0);
    recv(new_socket, B, sizeof(B), 0);

    // Matrix multiplication for one row
    for (int i = 0; i < SIZE; i++) {
        result[i] = 0;
        for (int j = 0; j < SIZE; j++) {
            result[i] += A_part[j] * B[j][i];
        }
    }

    // Send result
    send(new_socket, result, sizeof(result), 0);

    close(new_socket);
    close(server_fd);
    return 0;
}
