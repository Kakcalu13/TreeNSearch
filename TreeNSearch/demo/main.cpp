#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>
#include "TreeNSearch.h"
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>
#include <ctime>

const int N = 300;
float RADIUS = 0.12f;
int selected = 0;

std::vector<std::array<float, 3>> points;
std::vector<std::array<float, 2>> velocities;

void init_points() {
    srand(42);
    points.resize(N);
    velocities.resize(N);
    for (int i = 0; i < N; i++) {
        points[i][0] = ((float)rand()/RAND_MAX)*1.8f - 0.9f;
        points[i][1] = ((float)rand()/RAND_MAX)*1.8f - 0.9f;
        points[i][2] = 0.0f;
        float angle = ((float)rand()/RAND_MAX)*6.28f;
        float speed = 0.002f + ((float)rand()/RAND_MAX)*0.003f;
        velocities[i][0] = cosf(angle)*speed;
        velocities[i][1] = sinf(angle)*speed;
    }
}

void update_points() {
    for (int i = 0; i < N; i++) {
        points[i][0] += velocities[i][0];
        points[i][1] += velocities[i][1];
        if (points[i][0] >  0.95f) { points[i][0] =  0.95f; velocities[i][0] *= -1; }
        if (points[i][0] < -0.95f) { points[i][0] = -0.95f; velocities[i][0] *= -1; }
        if (points[i][1] >  0.95f) { points[i][1] =  0.95f; velocities[i][1] *= -1; }
        if (points[i][1] < -0.95f) { points[i][1] = -0.95f; velocities[i][1] *= -1; }
    }
}

void draw_circle(float cx, float cy, float r, int segs) {
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segs; i++) {
        float a = 2.0f*3.14159f*i/segs;
        glVertex2f(cx + r*cosf(a), cy + r*sinf(a));
    }
    glEnd();
}

void key_callback(GLFWwindow*, int key, int, int action, int) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_RIGHT) selected = (selected+1)%N;
        if (key == GLFW_KEY_LEFT)  selected = (selected-1+N)%N;
        if (key == GLFW_KEY_UP)    RADIUS = fminf(RADIUS+0.01f, 0.5f);
        if (key == GLFW_KEY_DOWN)  RADIUS = fmaxf(RADIUS-0.01f, 0.03f);
    }
}

int main() {
    init_points();

    if (!glfwInit()) return -1;
    GLFWwindow* win = glfwCreateWindow(800, 800, "TreeNSearch Demo - Arrow keys: cycle/resize", nullptr, nullptr);
    glfwMakeContextCurrent(win);
    glfwSetKeyCallback(win, key_callback);

    tns::TreeNSearch nsearch;
    const int set_0 = nsearch.add_point_set(points[0].data(), points.size());
    nsearch.set_active_search(set_0, set_0);

    while (!glfwWindowShouldClose(win)) {
        update_points();

        nsearch.set_search_radius(RADIUS);
        nsearch.run();

        glClearColor(0.08f, 0.08f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        const tns::NeighborList nl = nsearch.get_neighborlist(set_0, set_0, selected);

        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (int i = 0; i < N; i++) {
            const tns::NeighborList nl_i = nsearch.get_neighborlist(set_0, set_0, i);
            float t = fminf(nl_i.size() / 20.0f, 1.0f);
            glColor3f(t*0.2f, 0.3f + t*0.4f, 0.8f - t*0.5f);
            glVertex2f(points[i][0], points[i][1]);
        }
        glEnd();

        glColor3f(0.3f, 0.9f, 0.5f);
        glBegin(GL_LINES);
        for (int k = 0; k < nl.size(); k++) {
            int j = nl[k];
            glVertex2f(points[selected][0], points[selected][1]);
            glVertex2f(points[j][0], points[j][1]);
        }
        glEnd();

        glPointSize(7.0f);
        glColor3f(0.2f, 1.0f, 0.4f);
        glBegin(GL_POINTS);
        for (int k = 0; k < nl.size(); k++)
            glVertex2f(points[nl[k]][0], points[nl[k]][1]);
        glEnd();

        glColor3f(1.0f, 0.85f, 0.2f);
        draw_circle(points[selected][0], points[selected][1], RADIUS, 64);

        glPointSize(14.0f);
        glColor3f(1.0f, 0.25f, 0.25f);
        glBegin(GL_POINTS);
        glVertex2f(points[selected][0], points[selected][1]);
        glEnd();

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
