#ifndef VLR_COMMON_CAMERA
#define VLR_COMMON_CAMERA

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/glm.hpp>
#include <GL/glew.h>

namespace vlr
{
	namespace common
	{
		struct Viewport
		{
			int x, y, w, h;

			bool pointInViewport(int x, int y)
			{
				return x >= this->x && y >= this->y &&
					x < this->x + w && y < this->y + h;
			}
		};

		class Camera
		{
		public:
			Camera();
			Camera(const glm::vec3& position, const glm::quat& rotation);
			~Camera();

			void updateGL();

			inline glm::vec3 getPos() const;

			inline glm::mat4 getProjectionMatrix() const;
			inline glm::mat4 getRotationMatrix() const;

			inline glm::mat4 getMVP() const;
			
			inline glm::vec3 getLeft() const;
			inline glm::vec3 getUp() const;
			inline glm::vec3 getForward() const;

			inline Viewport getViewport() const;

			glm::vec3 worldSpaceToWorld(float x, float y,
				float dist) const;

			void setViewport(int x, int y, int width, int height);
			void perspective(float fov, float aspect, float near, float far);
			void orthographic(float scale, float aspect);

			void setPos(const glm::vec3& pos);
			void setRot(const glm::quat quat);

			void translate(const glm::vec3& offset);
			void rotate(const glm::quat& quat);

			void rotate(const glm::vec3& euler);

		protected:
			// Position & rotation
			glm::vec3 _position;
			glm::quat _rotation;

			// Viewport in screen space
			Viewport _viewport;

			// MVP matrices
			glm::mat4 _projection;
			glm::mat4 _rotationMatrix;
			glm::mat4 _view;
			glm::mat4 _model;

		private:
			Camera(const Camera&);
		};
		
		glm::vec3 Camera::getPos() const
		{
			return _position;
		}

		glm::mat4 Camera::getProjectionMatrix() const
		{
			return _projection;
		}

		glm::mat4 Camera::getRotationMatrix() const
		{
			return _rotationMatrix;
		}

		glm::mat4 Camera::getMVP() const
		{
			// Calculate MVP
			return _projection * _view * _model;
		}
		
		glm::vec3 Camera::getLeft() const
		{
			return glm::vec3(-1.0f, 0, 0) * _rotation;
		}
		
		glm::vec3 Camera::getUp() const
		{
			return glm::vec3(0, 1.0f, 0) * _rotation;
		}
		
		glm::vec3 Camera::getForward() const
		{
			return glm::vec3(0, 0, -1.0f) * _rotation;
		}

		Viewport Camera::getViewport() const
		{
			return _viewport;
		}
	}
}

#endif /* VLR_COMMON_CAMERA */
