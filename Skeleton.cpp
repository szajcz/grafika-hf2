//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szajcz Daniel
// Neptun : U9BUEF
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

const unsigned int screenWidth = 600, screenHeight = 600;

float EPS = 0.000001f;
struct vec3 {
	float x, y, z;

	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
	vec3 operator/(float a) const { return vec3(x / a, y / a, z / a); }

	vec3 operator+(const vec3& v) const {
		return vec3(x + v.x, y + v.y, z + v.z);
	}
	vec3 operator-(const vec3& v) const {
		return vec3(x - v.x, y - v.y, z - v.z);
	}
	vec3 operator*(const vec3& v) const {
		return vec3(x * v.x, y * v.y, z * v.z);
	}
	vec3 operator-() const {
		return vec3(-x, -y, -z);
	}
	vec3 normalize() const {
		return (*this) * (1 / (Length() + EPS));
	}
	float Length() const { return sqrtf(x * x + y * y + z * z); }

	operator float*() { return &x; }
};

float dot(const vec3& v1, const vec3& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};

struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
};

struct Complex {
	vec3 real;
	vec3 imaginary;

	Complex(vec3 r = vec3(0.0f, 0.0f, 0.0f), vec3 i = vec3(0.0f, 0.0f, 0.0f)) {
		real = r;
		imaginary = i;
	}
	vec3 complexAbs() {
		return vec3(sqrt(real.x * real.x + imaginary.x * imaginary.x), sqrt(real.y * real.y + imaginary.y * imaginary.y), sqrt(real.z * real.z + imaginary.z * imaginary.z));
	}
	Complex operator+(Complex c) {
		return Complex(vec3(real.x + c.real.x, real.y + c.real.y, real.z + c.real.z), vec3(imaginary.x + c.imaginary.x, imaginary.y + c.imaginary.y, imaginary.z + c.imaginary.z));
	}

	Complex operator-(Complex c) {
		return Complex(vec3(real.x - c.real.x, real.y - c.real.y, real.z - c.real.z), vec3(imaginary.x - c.imaginary.x, imaginary.y - c.imaginary.y, imaginary.z - c.imaginary.z));
	}
	Complex operator*(Complex c)	{
		Complex a;

		a.real.x = (real.x*c.real.x) - (imaginary.x*c.imaginary.x);
		a.real.y = (real.y*c.real.y) - (imaginary.y*c.imaginary.y);
		a.real.z = (real.z*c.real.z) - (imaginary.z*c.imaginary.z);

		a.imaginary.x = (real.x*c.imaginary.x) + (imaginary.x*c.real.x);
		a.imaginary.y = (real.y*c.imaginary.y) + (imaginary.y*c.real.y);
		a.imaginary.z = (real.z*c.imaginary.z) + (imaginary.z*c.real.z);

		return a;
	}
	Complex operator/(Complex c)
	{
		Complex a;

		a.real.x = ((real.x*c.real.x) + (imaginary.x*c.imaginary.x)) / ((c.real.x*c.real.x) + (c.imaginary.x*c.imaginary.x));
		a.real.y = ((real.y*c.real.y) + (imaginary.y*c.imaginary.y)) / ((c.real.y*c.real.y) + (c.imaginary.y*c.imaginary.y));
		a.real.z = ((real.z*c.real.z) + (imaginary.z*c.imaginary.z)) / ((c.real.z*c.real.z) + (c.imaginary.z*c.imaginary.z));

		a.imaginary.x = ((imaginary.x*c.real.x) - (real.x*c.imaginary.x)) / ((c.real.x*c.real.x) + (c.imaginary.x*c.imaginary.x));
		a.imaginary.y = ((imaginary.y*c.real.y) - (real.y*c.imaginary.y)) / ((c.real.y*c.real.y) + (c.imaginary.y*c.imaginary.y));
		a.imaginary.z = ((imaginary.z*c.real.z) - (real.z*c.imaginary.z)) / ((c.real.z*c.real.z) + (c.imaginary.z*c.imaginary.z));

		return a;
	}
};

class Material {
	vec3 k;

	// smooth material elements
	vec3 F0;
	vec3 n;

	// rough material elements
	vec3 ks, kd;
	float shininess;

public:
	bool reflective, refractive, rough;

	Material() {}
	Material(vec3 n_in, vec3 k_in, float shininess_in, bool refl, bool refr, bool r, vec3 ks_in, vec3 kd_in) {
		k = k_in;
		n = n_in;

		ks = ks_in;
		kd = kd_in;
		shininess = shininess_in;

		reflective = refl;
		refractive = refr;
		rough = r;
	}

	vec3 reflect(vec3 inDir, vec3 normal) {
		return inDir - normal * dot(normal, inDir) * 2.0f;
	}

	vec3 refract(vec3 inDir, vec3 normal) {
		float ior = n.x;
		float cosa = -dot(normal, inDir);
		if (cosa < 0) { cosa = -cosa; normal = -normal; ior = 1 / n.x; }
		float disc = 1 - (1 - cosa * cosa) / ior / ior;
		if (disc < 0) return reflect(inDir, normal);
		return inDir / ior + normal * (cosa / ior - sqrt(disc));
	}

	vec3 Fresnel(vec3 inDir, vec3 normal) {
		float cosTheta = fabs(dot(normal, inDir));
		Complex complexCosTheta = Complex(vec3(cosTheta, cosTheta, cosTheta), vec3(0, 0, 0));

		Complex complexPart = Complex(n, k);
		
		Complex complexCosThetaComma = complexCosTheta;

		vec3 result = (((complexCosTheta - complexPart * complexCosThetaComma) / (complexCosTheta + complexPart * complexCosThetaComma)).complexAbs() *
			((complexCosTheta - complexPart * complexCosThetaComma) / (complexCosTheta + complexPart * complexCosThetaComma)).complexAbs()) * 0.5f
			+ (((complexCosThetaComma - complexPart * complexCosTheta) / (complexCosThetaComma + complexPart * complexCosTheta)).complexAbs() *
			((complexCosThetaComma - complexPart * complexCosTheta) / (complexCosThetaComma + complexPart * complexCosTheta)).complexAbs()) * 0.5f;
		return result;		
	}

	vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 inRad) {
		vec3 reflRad(0, 0, 0);
		float cosTheta = dot(normal, lightDir);
		if (cosTheta < 0) return reflRad;
		reflRad = inRad * kd * cosTheta;
		vec3 halfway = (viewDir + lightDir).normalize();
		float cosDelta = dot(normal, halfway);
		if (cosDelta < 0) return reflRad;
		return reflRad + inRad * ks * pow(cosDelta, shininess);
	}

	vec3 getTextured(vec3 point) {
		int square = (int)floor(point.x);
		if ((square % 2) == 0) {
			return vec3(1.0f, 0.0f, 0.0f);
		}
		return vec3(1.0f, 1.0f, 1.0f);
	}
};

struct Hit {
	float t;
	vec3 position;
	vec3 normal;
	Material * material;
	Hit() { t = -1; }
	Hit(float t_in, vec3 p_in, vec3 normal_in, Material* mat_in) {
		t = t_in;
		position = p_in;
		normal = normal_in;
		material = mat_in;
	}
};

struct Ray {
	vec3 source, dir;
	Ray(vec3 _start, vec3 _dir) { source = _start; dir = _dir.normalize(); }
};


class Intersectable {
protected:
	Material * mat;
public:
	virtual Hit findIntersection(Ray r) = 0;
};

class Plane : public virtual Intersectable {
	vec3 normal, center;

public:
	Plane(vec3 center_in, vec3 norm_in, Material* mat_in) {
		center = center_in;
		normal = norm_in;
		mat = mat_in;
	}

	vec3 Normal(vec3 x) {
		return normal;
	}

	Hit findIntersection(Ray r) {
		float d = dot(normal, r.dir);
		if (fabs(d) > EPS) {
			float t = dot((center - r.source), normal) / d;
			if (t >= EPS) return Hit(t, r.source + r.dir * t, normal, mat);
		}
		return Hit();
	}
};

class Cylinder : public virtual Intersectable {
public:
	vec3 pointA, pointB;
	float radius;

	Cylinder(vec3 a_in, vec3 b_in, float rad_in, Material * mat_in) :pointA(a_in), pointB(b_in), radius(rad_in) {
		mat = mat_in;
	}

	vec3 Normal(vec3 x) {
		vec3 AtoP = x - pointA;
		vec3 AtoB = pointB - pointA;
		vec3 best = pointA + AtoB * (dot(AtoP, AtoB) / dot(AtoB, AtoB));
		vec3 normal = (x - best).normalize();
		return normal * -1; // *-1 because we are inside
	}

	Hit findIntersection(Ray ray) {
		// cylinder - ray intersection
		vec3 AtoB = pointB - pointA;
		vec3 AtoRS = ray.source - pointA;
		vec3 ARScrossAB = cross(AtoRS, AtoB);
		vec3 RDcrossAB = cross(ray.dir, AtoB);
		float ab2 = dot(AtoB, AtoB);
		float a = dot(RDcrossAB, RDcrossAB);
		float b = 2 * dot(RDcrossAB, ARScrossAB);
		float c = dot(ARScrossAB, ARScrossAB) - (radius * radius * ab2);

		float determinant = powf(b, 2) - 4 * a * c;

		if (determinant < 0) {
			return Hit();
		}
		float t1 = (-b + sqrt(determinant)) / (2 * a);
		float t2 = (-b - sqrt(determinant)) / (2 * a);

		vec3 point;
		if (t1 > t2) {
			point = ray.source + ray.dir * t1;
			return Hit(t1, point, Normal(point), mat);
		}
		else {
			point = ray.source + ray.dir * t2;
			return Hit(t2, point, Normal(point), mat);
		}
		return Hit();
	}
};

class Triangle : public virtual Intersectable {
	vec3 p1, p2, p3, normal;

public:
	Triangle(Material * mat_in, vec3& p1, vec3& p2, vec3& p3) :
		p1(p1), p2(p2), p3(p3), normal(cross((p2 - p1).normalize(), (p3 - p1).normalize())) {
		mat = mat_in;
	}

	vec3 Normal(vec3 x) {
		return normal;
	}

	Hit findIntersection(Ray r) {
		if (dot(r.dir, normal) == 0) return Hit();
		float t = dot((p1 - r.source), normal) / dot(r.dir, normal);
		if (t < 0) return Hit();

		vec3 p = r.source + r.dir * t;

		if ((dot(cross(p2 - p1, p - p1), normal) > 0) &&
			(dot(cross(p3 - p2, p - p2), normal) > 0) &&
			(dot(cross(p1 - p3, p - p3), normal) > 0)) {
			return Hit(t, p, normal, mat);
		}
		return Hit();
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float f = w.Length();
		right = cross(vup, w).normalize() * f * tan(fov / 2);
		up = cross(w, right).normalize() * f * tan(fov / 2);
	}
	Ray getray(int X, int Y) {
		vec3 dir = lookat + right * (2.0 * (X + 0.5) / screenWidth - 1) + up * (2.0 * (Y + 0.5) / screenHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

class Light {
public:
	vec3 direction;
	vec3 color;
	float intensity;

	Light() {}
	Light(vec3 dir_in, vec3 color_in, float int_in) {
		direction = dir_in;
		color = color_in;
		intensity = int_in;
	}

	vec3 Le(vec3 hitPos) {
		float dist = (hitPos - direction).Length();
		return (color / dist) * (intensity / dist);
	}
};

class Scene {
	std::vector<Intersectable *> objects;
	Light lights[3];
	Camera camera;
	vec3 La;
public:

	vec3 torusParametric(float u, float v, float R, float r, bool refl) {
		if (refl) { // instead of the sgn
			return vec3((R + r * cosf(v)) * cosf(u), r * sinf(v), (R + r * cosf(v)) * sinf(u));
		}
		return vec3((R + r * cosf(v)) * cosf(u), (R + r * cosf(v)) * sinf(u), r * sinf(v));
	}

	vec3 rotateVectorXYZ(vec3 v3, float alfa_x, float beta_y, float gamma_z) {
		vec4 v4(v3.x, v3.y, v3.z, 0);

		mat4 MrotateX(
			1, 0, 0, 0,
			0, cosf(alfa_x), sinf(alfa_x), 0,
			0, -sinf(alfa_x), cosf(alfa_x), 0,
			0, 0, 0, 1);

		mat4 MrotateY(
			cosf(beta_y), 0, -sinf(beta_y), 0,
			0, 1, 0, 0,
			sinf(beta_y), 0, cosf(beta_y), 0,
			0, 0, 0, 1);

		mat4 MrotateZ(
			cosf(gamma_z), sinf(gamma_z), 0, 0,
			-sinf(gamma_z), cosf(gamma_z), 1, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		vec4 rotated = v4 * MrotateX * MrotateY * MrotateZ;
		return vec3(rotated.v[0], rotated.v[1], rotated.v[2]);
	}

	void buildTorus(Material* mat, vec3 pos, float alfa, float beta, float gamma, bool refl) {
		float R = 0.5f;
		float r = 0.1f;
		int stepSizeU = 5;
		int stepSizeV = 5;
		float stepU = M_PI * 2 / stepSizeU;
		float stepV = M_PI * 2 / stepSizeV;

		Triangle *t1;
		Triangle *t2;
		for (int u = 0; u < stepSizeU; u++) {
			for (int v = 0; v < stepSizeV; v++) {
				vec3 point1 = torusParametric(u*stepU, v*stepV, R, r, refl);
				point1 = rotateVectorXYZ(point1, alfa, beta, gamma);
				point1 = point1 + pos;

				vec3 point2 = torusParametric((u + 1)*stepU, v*stepV, R, r, refl);
				point2 = rotateVectorXYZ(point2, alfa, beta, gamma);
				point2 = point2 + pos;

				vec3 point3 = torusParametric(u*stepU, (v + 1)*stepV, R, r, refl);
				point3 = rotateVectorXYZ(point3, alfa, beta, gamma);
				point3 = point3 + pos;

				vec3 point4 = torusParametric((u + 1)*stepU, (v + 1)*stepV, R, r, refl);
				point4 = rotateVectorXYZ(point4, alfa, beta, gamma);
				point4 = point4 + pos;

				t1 = new Triangle(mat, point1, point3, point2);
				t2 = new Triangle(mat, point4, point2, point3);
				objects.push_back(t1);
				objects.push_back(t2);
			}
		}
	}

	void build() {
		// setting up the camera
		camera.set(vec3(0, 0, 3), vec3(0, 0, 0), vec3(0, 1, 0), 45 * M_PI / 180);

		// lights
		La = vec3(0.1f, 0.1f, 0.1f);
		lights[0] = Light(vec3(0.0f, 2.0f, -2.0f), vec3(0.9f, 0.9f, 0.5f), 25); // yellowish
		lights[1] = Light(vec3(1.0f, -2.0f, -1.0f), vec3(0.0f, 0.0f, 1.0f), 20); // blue
		lights[2] = Light(vec3(-1.5f, 0.0f, -2.0f), vec3(0.6f, 1.0f, 0.6f), 15); // light green

		// material
		Material* textured = new Material(vec3(0, 0, 0), vec3(0, 0, 0), 20.0f, false, false, true, vec3(1.0f, 1.0f, 1.0f), vec3(0.2f, 0.2f, 0.2f));
		Material* gold = new Material(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f), 10, true, false, false, vec3(0, 0, 0), vec3(0, 0, 0));
		Material* silver = new Material(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f), 15, true, false, false, vec3(0, 0, 0), vec3(0, 0, 0));
		Material* glass = new Material(vec3(1.5f, 1.5f, 1.5f), vec3(0.0f, 0.0f, 0.0f), 0, false, true, false, vec3(0, 0, 0), vec3(0, 0, 0));

		// building the room
		Plane* ground = new Plane(vec3(0.0f, -2.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), textured);
		Plane* ceiling = new Plane(vec3(0.0f, 2.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f), textured);
		Cylinder* wall = new Cylinder(vec3(0,0,0), vec3(4,0,0), 4.5f, textured);
		objects.push_back(ground);
		objects.push_back(ceiling);
		objects.push_back(wall);

		// toruses
		buildTorus(silver, vec3(0.1f, 0.0f, -2.0f), 20, 12, 30, true);
		buildTorus(gold, vec3(0.2f, 0.5f, -1.8f), 40, 12, 30, true);
		buildTorus(glass, vec3(0.0f, -0.5f, -1.5f), 10, 70, 10, false);

	}

	void render(vec3 image[]) {
#pragma omp parallel for
		for (int Y = 0; Y < screenHeight; Y++) {
			for (int X = 0; X < screenWidth; X++) image[Y * screenWidth + X] = trace(camera.getray(X, Y), 0);
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->findIntersection(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		return bestHit;
	}

	int sign(float x) {
		if (x > 0) return 1;
		if (x < 0) return -1;
		return 0;
	}

	int maxDepth = 10;
	vec3 trace(Ray ray, int depth) {
		if (depth > maxDepth) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La; // nothing

		vec3 outRadiance;
		if (hit.material->rough) {
			outRadiance = hit.material->getTextured(hit.position) * La;
			for (int i = 0; i < 3; i++) {
				Light l = lights[i];
				vec3 toLightDir = (l.direction - hit.position).normalize();
				Ray shadowRay(hit.position + (hit.normal * EPS),toLightDir);
				Hit shadowHit = firstIntersect(shadowRay);
				if (shadowHit.t < 0 || shadowHit.t >(hit.position - l.direction).Length()) {
					outRadiance = outRadiance + hit.material->shade(hit.normal, ray.dir, toLightDir, l.Le(hit.position));
				}
			}
		}
		if (hit.material->reflective) {
			vec3 reflectionDir = hit.material->reflect(ray.dir, hit.normal);
			Ray reflectedRay(hit.position + (hit.normal * EPS),reflectionDir);
			outRadiance = outRadiance + trace(reflectedRay, depth + 1) * hit.material->Fresnel(ray.dir, hit.normal);
		}
		if (hit.material->refractive) {
			vec3 refractionDir = hit.material->refract(ray.dir, hit.normal).normalize();
			Ray refractedRay(hit.position - (hit.normal * EPS * sign(dot(hit.normal, ray.dir))), refractionDir);
			outRadiance = outRadiance + trace(refractedRay, depth + 1)*(vec3(1, 1, 1) - hit.material->Fresnel(ray.dir, hit.normal));
		}
		return outRadiance;
	}
};

Scene scene;

void getErrorInfo(unsigned int handle) {
	int logLen, written;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";


// handle of the shader program
unsigned int shaderProgram;

class FullScreenTexturedQuad {
	unsigned int vao, textureId;	// vertex array object id and texture id
public:
	void Create(vec3 image[]) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

								// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

																	  // Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screenWidth, screenHeight, 0, GL_RGB, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 6);	// draw two triangles forming a quad
	}
};

// The virtual world: single quad
FullScreenTexturedQuad fullScreenTexturedQuad;
vec3 image[screenWidth * screenHeight];	// The image, which stores the ray tracing result

										// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, screenWidth, screenHeight);
	scene.build();
	scene.render(image);
	fullScreenTexturedQuad.Create(image);

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	glUseProgram(shaderProgram); 	// make this program run
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

